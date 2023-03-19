import os
from pathlib import Path

code_filetypes = {".py", ".html", ".json", ".h", ".cpp"}
excluded_filetypes = {".csv", ".txt", ".md"}

preamble = "I wrote a game where users solve puzzles to score points,\
and the puzzles are simple math equations using the vector embeddings \
of words, like nouns and verbs. Each puzzle has some random words displayed \
 separated by addition signs, which are added together using the vector \
embedding of those words producing a resultant vector or 'correct answer'.\
 A user submits a word or several words that try to be close to the resultant \
answer in the sense that the vector addition of the user's input words are close\
 to the resultant vector. The code is organized as a python project that deploys\
 via flask to a html website. There is an 'app.py' file and a 'templates/index.html' \
file as well as files that contain words like nouns, synonyms, antonyms, and adjectives located \
inside a 'wordlists' folder  and saved as .txt files.  You are going to develop some new \
features for the game after reading the existing codebase. The code is pasted \
below with the name of the file and then the code inside in markdown \
format. Let me know when you are ready."

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


summary_markdown = [preamble + "# GitHub Repo Summary\n"]
summary_markdown.extend(generate_summary_markdown(os.getcwd()))

with open("github_repo_summary.md", "w") as summary_file:
    summary_file.writelines(summary_markdown)

print("GitHub Repo Summary generated in github_repo_summary.md")
