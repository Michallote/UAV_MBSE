import os


def create_prompt_file(file_list: str, output_file="prompt.txt"):
    """
    Takes a list of file paths and creates a prompt file for LLM processing.

    Args:
        file_list (list of str): List of file paths to read and document.
        output_file (str): Path to save the generated prompt file.

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as prompt_file:
        for file_path in file_list:
            # Ensure the file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Get the file name for the header
            file_name = os.path.basename(file_path)

            # Write the header with the file name
            prompt_file.write(f"File: {file_name}\n")
            prompt_file.write("=" * (len(file_name) + 6) + "\n\n")

            # Read the file content and write it to the prompt file
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    prompt_file.write(content)
                    prompt_file.write("\n\n")  # Add extra spacing between files
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                prompt_file.write(f"Error reading {file_path}: {e}\n\n")

    print(f"Prompt file created at {output_file}")


# Example usage:
if __name__ == "__main__":

    file_list = [
        "main.py",
        "src/structures/structural_model.py",
        "src/structures/spar.py",
        "src/structures/inertia_tensor.py",
        "src/utils/intersection.py",
    ]

    create_prompt_file(file_list, "structures_summary.txt")


def list_files(directory: str) -> list:
    """
    List all files in a directory.

    Args:
        directory (str): Path to the directory to list files from.

    Returns:
        list: List of file paths in the directory.
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]


list_files("src/utils")
