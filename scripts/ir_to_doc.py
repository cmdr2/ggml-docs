import json
import os
import sys
import argparse


def load_json(filename):
    """Load and return JSON data from a file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using empty dictionary/list.")
        return {} if filename != "categories.json" else []


def create_anchor(text):
    """Create a markdown-compatible anchor from text"""
    return text.lower().replace("_", "-")


def generate_functions_md(functions, categories, descriptions, link_prefix=""):
    """Generate the functions markdown content"""
    content = ["# Functions\n\n## Index\n"]

    # First, create the categorized list of functions
    for category in categories:
        content.append("<details>")
        content.append(f"<summary>{category['name']}</summary>\n")
        for func_name in category["functions"]:
            line = ""
            if func_name in functions:
                func = functions[func_name]
                line = f"- [{func['signatureShort']}](#{create_anchor(func_name)})"
                if func_name in descriptions:
                    line += f" - {descriptions[func_name]}"

            content.append(line)
        content.append("</details>")
        content.append("\n")

    # Then create the detailed definitions
    content.append("## Detailed Definitions\n")
    for category in categories:
        for func_name in category["functions"]:
            lines = []
            if func_name in functions:
                func = functions[func_name]
                lines.append(f"### {func_name}\n")

                if func_name in descriptions:
                    lines.append(f"{descriptions[func_name]}\n")

                lines.append("```c")
                lines.append(func["signature"])
                lines.append("```\n")
                lines.append(f"Source: [{func['location']}]({link_prefix}{func['location']})\n")

            content += lines

    return "\n".join(content)


def generate_type_md(title, items, link_prefix=""):
    """Generate markdown content for enums, typedefs, or structs"""
    content = [f"# {title}\n\n## Index\n"]

    # First, create the index
    for name, item in items.items():
        content.append(f"- [{name}](#{create_anchor(name)})")

    content.append("\n")

    # Then create the detailed definitions
    content.append("## Detailed Definitions\n")
    for name, item in items.items():
        content.append(f"### {name}\n")
        content.append("```c")
        content.append(item["definition"])
        content.append("```\n")
        content.append(f"Source: [{item['location']}]({link_prefix}{item['location']})\n")

    return "\n".join(content)


def main():
    parser = argparse.ArgumentParser(description="Convert intermediate JSON files to markdown documentation")
    parser.add_argument(
        "--source-dir", help="Path to the intermediate representation directory (with contains the JSON files)"
    )
    parser.add_argument("--output-dir", help="Path to the output directory")
    parser.add_argument("--link-prefix", help="Prefix to apply before links to ggml's code")
    args = parser.parse_args()

    # Load all JSON files
    functions = load_json(f"{args.source_dir}/functions.json")
    descriptions = load_json(f"{args.source_dir}/descriptions.json")
    enums = load_json(f"{args.source_dir}/enums.json")
    typedefs = load_json(f"{args.source_dir}/typedefs.json")
    structs = load_json(f"{args.source_dir}/structs.json")
    categories = load_json(f"{args.source_dir}/categories.json")

    # Create output directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)

    # Generate and write functions.md
    with open(f"{args.output_dir}/functions.md", "w") as f:
        f.write(generate_functions_md(functions, categories, descriptions, link_prefix=args.link_prefix))

    # Generate and write other documentation files
    type_files = {
        "enums.md": ("Enums", enums),
        "typedefs.md": ("Typedefs", typedefs),
        "structs.md": ("Structs", structs),
    }

    for filename, (title, data) in type_files.items():
        with open(f"{args.output_dir}/{filename}", "w") as f:
            f.write(generate_type_md(title, data, link_prefix=args.link_prefix))


if __name__ == "__main__":
    main()
