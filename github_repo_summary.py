import os
from pathlib import Path

code_filetypes = {".py", ".html", ".json", ".h", ".cpp"}
excluded_filetypes = {".csv", ".txt", ".md"}

def generate_summary_markdown(directory, repo_path=""):
    markdown_lines = []

    for entry in os.scandir(directory):
        if entry.is_dir():
            subdir_path = os.path.join(repo_path, entry.name)
            markdown_lines.extend(generate_summary_markdown(entry.path, subdir_path))
        else:
            file_ext = Path(entry.path).suffix
            relative_path = os.path.join(repo_path, entry.name)

            if (
                file_ext in code_filetypes
                and file_ext not in excluded_filetypes
                and entry.name != "github_repo_summary.py"
            ):
                markdown_lines.append(f"## {relative_path}\n")
                markdown_lines.append("```" + file_ext.lstrip(".") + "\n")
                with open(entry.path, "r") as file:
                    file_content = file.read()
                    markdown_lines.append(file_content)
                markdown_lines.append("```\n")

    return markdown_lines


summary_markdown = ["# GitHub Repo Summary\n"]
summary_markdown.extend(generate_summary_markdown(os.getcwd()))

with open("github_repo_summary.md", "w") as summary_file:
    summary_file.writelines(summary_markdown)

print("GitHub Repo Summary generated in github_repo_summary.md")
