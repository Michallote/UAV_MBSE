# UAV_MBSE
An application for developing aircraft plotting in python!

This app parses XML aircraft exported from XFLR5 and turns the definitions into 3D data upon which numerical calculations and estimations can be performed. 


        "editor.formatOnPaste": true,
        "editor.formatOnSave": true,
        "editor.formatOnType": true,
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },



keybindings.json

    {
        "key": "ctrl+l",
        "command": "settings.cycle",
        "when": "editorTextFocus",
        "args": {
            "id": "relativeLineNumbers",
            "values": [
                {
                    "editor.lineNumbers": "on"
                },
                {
                    "editor.lineNumbers": "relative"
                }
            ]
        }
    }


pickler

import pickle

def save_to_pkl(input_file_path, output_pkl_path):
    try:
        # Read the binary file
        with open(input_file_path, 'rb') as file:
            binary_data = file.read()
        
        # Serialize and save the binary data to a .pkl file
        with open(output_pkl_path, 'wb') as pkl_file:
            pickle.dump(binary_data, pkl_file)
        
        print(f"Data from {input_file_path} has been saved to {output_pkl_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file_path = 'example_input.bin'  # Path to your input binary file
output_pkl_path = 'output_data.pkl'    # Path to save the .pkl file
save_to_pkl(input_file_path, output_pkl_path)

