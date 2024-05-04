import os


def rename_files(directory):
    # List all files in the provided directory
    files = os.listdir(directory)

    for filename in files:
        # Ensure it's a file, not a directory or subdirectory
        if os.path.isfile(os.path.join(directory, filename)):
            # Split the filename from its extension
            name_part, extension_part = os.path.splitext(filename)

            # Replace the dot in the numeric values with 'p', only affecting the filename part
            new_name_part = name_part.replace(".", "p").replace("\x81", "")

            if new_name_part != name_part:
                # Combine the new filename part with its original extension
                new_filename = new_name_part + extension_part

                # Full path for old and new filenames
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(old_file, new_file)
                print(f"Renamed '{filename}' to '{new_filename}'")


# Example usage
directory = r"E:\Documentos\Aero Design\ProgramaMaestro-20211019T045941Z-001\ProgramaMaestro\BasesDatosR\polars"
rename_files(directory)


match = "GOE 235"
files = os.listdir(directory)
gen = (file for file in files if match in file)
filename = next(gen)
filename.casefold()
