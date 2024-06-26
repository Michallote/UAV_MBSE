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
```
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
```

depickler

```
import pickle

def restore_from_pkl(input_pkl_path, output_file_path):
    try:
        # Load the binary data from the .pkl file
        with open(input_pkl_path, 'rb') as pkl_file:
            binary_data = pickle.load(pkl_file)
        
        # Write the binary data back to a new file
        with open(output_file_path, 'wb') as file:
            file.write(binary_data)
        
        print(f"Data has been restored from {input_pkl_path} to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_pkl_path = 'output_data.pkl'    # Path to the .pkl file containing the serialized data
output_file_path = 'restored_output.bin'  # Path for saving the restored file
restore_from_pkl(input_pkl_path, output_file_path)

```

![image](https://github.com/Michallote/UAV_MBSE/assets/74160122/f8bfe71c-973f-40be-8651-1411cfbb8c59)

