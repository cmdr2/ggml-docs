import os
import json
import argparse
from typing import Dict, List
from clang.cindex import Index, CursorKind, Cursor


class ClangCodeAnalyzer:
    def __init__(self):
        self.index = Index.create()

    def get_cursor_location(self, cursor: Cursor, base_path: str) -> str:
        """Get the relative file location with line number for a cursor."""
        if cursor.location.file:
            file_path = os.path.relpath(cursor.location.file.name, base_path)
            return f"{file_path}#L{cursor.location.line}"
        return ""

    def get_function_signature(self, cursor: Cursor, short: bool = False) -> str:
        """Extract the complete function signature."""
        result_type = cursor.result_type.spelling
        name = cursor.spelling
        args = []

        for arg in cursor.get_arguments():
            arg_txt = arg.spelling if short else f"{arg.type.spelling} {arg.spelling}"
            args.append(arg_txt)

        fn_txt = f"{name}({', '.join(args)})" if short else f"{result_type} {name}({', '.join(args)})"
        return fn_txt

    def get_struct_definition(self, cursor: Cursor) -> str:
        """Extract the struct definition."""
        if len(list(cursor.get_children())) == 0:
            # Empty struct
            return f"struct {cursor.spelling};"

        fields = []
        for field in cursor.get_children():
            if field.kind == CursorKind.FIELD_DECL:
                fields.append(f"    {field.type.spelling} {field.spelling};")

        return f"struct {cursor.spelling} {{\n{chr(10).join(fields)}\n}};"

    def get_enum_definition(self, cursor: Cursor) -> str:
        """Extract the enum definition."""
        values = []
        for enum_value in cursor.get_children():
            if enum_value.kind == CursorKind.ENUM_CONSTANT_DECL:
                value_str = enum_value.spelling
                if enum_value.enum_value is not None:
                    value_str += f" = {enum_value.enum_value}"
                values.append(f"    {value_str}")

        return f"enum {cursor.spelling} {{\n{chr(10).join(values)}\n}};"

    def analyze_file(self, file_path: str, base_path: str) -> Dict[str, List[Dict[str, str]]]:
        """Analyze a single C file using libclang."""
        result = {"functions": {}, "structs": {}, "typedefs": {}, "enums": {}}

        # Parse the file
        tu = self.index.parse(file_path)
        if not tu:
            print(f"Failed to parse {file_path}")
            return result

        def visit_node(node: Cursor):
            # Skip system headers
            if node.location.file and node.location.file.name != file_path:
                return

            if node.spelling.startswith("(unnamed"):
                node.spelling = get_uuid()

            if node.kind == CursorKind.FUNCTION_DECL:
                result["functions"][node.spelling] = {
                    "signature": self.get_function_signature(node),
                    "signatureShort": self.get_function_signature(node, short=True),
                    "location": self.get_cursor_location(node, base_path),
                }

            elif node.kind == CursorKind.STRUCT_DECL and node.spelling:
                result["structs"][node.spelling] = {
                    "definition": self.get_struct_definition(node),
                    "location": self.get_cursor_location(node, base_path),
                }

            elif node.kind == CursorKind.TYPEDEF_DECL:
                underlying_type = node.underlying_typedef_type.spelling
                result["typedefs"][node.spelling] = {
                    "definition": f"typedef {underlying_type} {node.spelling};",
                    "location": self.get_cursor_location(node, base_path),
                }

            elif node.kind == CursorKind.ENUM_DECL and node.spelling:
                result["enums"][node.spelling] = {
                    "definition": self.get_enum_definition(node),
                    "location": self.get_cursor_location(node, base_path),
                }

            # Recursively visit children
            for child in node.get_children():
                visit_node(child)

        visit_node(tu.cursor)
        return result

    def scan_directory(self, source_path: str) -> Dict[str, List[Dict[str, str]]]:
        """Recursively scan directory for C files and analyze them."""
        result = {"functions": {}, "structs": {}, "typedefs": {}, "enums": {}}

        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith((".c", ".h")):
                    file_path = os.path.join(root, file)
                    file_result = self.analyze_file(file_path, source_path)

                    # Merge results
                    for key in result:
                        result[key].update(file_result[key])

        return result


def main():
    parser = argparse.ArgumentParser(description="Analyze C source files using libclang and generate JSON output")
    parser.add_argument("--source-dir", help="Path to the source directory")
    parser.add_argument("--output-dir", help="Path to the output directory")
    args = parser.parse_args()

    analyzer = ClangCodeAnalyzer()
    result = analyzer.scan_directory(args.source_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Write results to JSON file
    for key in ("functions", "structs", "typedefs", "enums"):
        res = result[key]
        out_file = os.path.join(args.output_dir, f"{key}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


def get_uuid():
    import uuid

    # Generate a UUID
    generated_uuid = uuid.uuid4()

    # Get the first 8 characters
    return str(generated_uuid)[:8]


if __name__ == "__main__":
    main()


"""
# Prompt for generating function descriptions (reasoning+search):
    Group these functions into categories. Write a JSON file with a list of objects, where each object is a category. The category object has a name key, a title field, and a 'functions' key, which contains a list of function names in that category. Example names for categories: "Reduction & Repetition Operations", "Element–wise Arithmetic & Accumulation", "Tensor Duplication, Copy & Reshaping" etc.

    tallocr is "Tensor Allocator" and gallocr is "Graph allocator"

    Move functions related to context management into a separate category.

    These functions are in https://github.com/ggml-org/ggml and used in llama.cpp

    <include the list of function names (without signatures)>

# Prompt for categorizing the functions (reasoning+search):
    Here's a list of functions in ggml. Write a one line description for each function, and produce a JSON with the function name as the key, and description as the value.

    These functions are in https://github.com/ggml-org/ggml and used in llama.cpp

    <include the list of function names (without signatures)>
"""
