# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:08:12 2023

@author: Michel Gordillo
"""




def set_preamble(name):
    file_name = "Attribute VB_Name = \"{}\"\n".format(name)
    
    file_contents = (
        file_name+
        "' ******************************************************************************\n"
        "'Macro recorded on 02/02/23 by Michel Gordillo\n"
        "' ******************************************************************************\n"
        "Dim swApp As Object\n"
        "\n"
        "Dim Part As Object\n"
        "Dim boolstatus As Boolean\n"
        "Dim longstatus As Long, longwarnings As Long\n"
        "\n"
        "Sub main()\n"
        "\n"
        "Set swApp = Application.SldWorks\n"
        "\n"
        "Set Part = swApp.ActiveDoc\n"
        "Dim COSMOSWORKSObj As Object\n"
        "Dim CWAddinCallBackObj As Object\n"
        "Set CWAddinCallBackObj = swApp.GetAddInObject(\"CosmosWorks.CosmosWorks\")\n"
        "Set COSMOSWORKSObj = CWAddinCallBackObj.COSMOSWORKS\n"
        "Dim myModelView As Object\n"
        "Set myModelView = Part.ActiveView\n"
        "myModelView.FrameState = swWindowState_e.swWindowMaximized\n"
        "\n"
    )
    return file_contents

def set_end_code():
    return (
        "Set CWAddinCallBackObj = Nothing\n"
        "Set COSMOSWORKSObj = Nothing\n"
        "End Sub\n")
    
def insert_curve_file(coordinates,newname):
    
    global CURVE_COUNTER
    CURVE_COUNTER +=  1   
    name = 'Curve'+str(CURVE_COUNTER)
    command_head = ("Part.InsertCurveFileBegin\n")
    
    command_tail = (
        "boolstatus = Part.InsertCurveFileEnd()\n"
        "\n"
        "boolstatus = Part.Extension.SelectByID2(\"{}\", \"REFERENCECURVES\", 0, 0, 0, False, 0, Nothing, 0)\n".format(name)+
        "Part.ClearSelection2 True\n"
        "boolstatus = Part.Extension.SelectByID2(\"{}\", \"REFERENCECURVES\", 0, 0, 0, False, 0, Nothing, 0)\n".format(name)+
        "boolstatus = Part.SelectedFeatureProperties(0, 0, 0, 0, 0, 0, 0, 1, 0, \"{}\")\n").format(newname)


    insert_coordinate_lines =''
    
    for coordinate in coordinates:
        insert_coordinate_lines += "boolstatus = Part.InsertCurveFilePoint({},{},{})\n".format(*coordinate)
        
    return command_head + insert_coordinate_lines + command_tail

def insert_feature_tree_folder(name_list, folder_name):
    
    command_lines = "Part.ClearSelection2 True\n"
    n = len(name_list)
    appended_elements = [False if i in [0, n-1] else True for i in range(n)]
    
    for name,appended in zip(name_list,appended_elements):
        command_lines += "boolstatus = Part.Extension.SelectByID2(\"{}\", \"REFERENCECURVES\", 0, 0, 0, {}, 0, Nothing, 0)\n".format(name,appended)
    
    command_tail = (
        "Dim myFeature As Object\n"
        "Set myFeature = Part.FeatureManager.InsertFeatureTreeFolder2(swFeatureTreeFolderType_e.swFeatureTreeFolder_Containing)\n"
        "boolstatus = Part.SelectedFeatureProperties(0, 0, 0, 0, 0, 0, 0, 1, 0, \"{}\")\n".format(folder_name))
    
    command_lines += command_tail
    
    return (command_lines)

def insert_curves_features(coordinates_lists,name_list,folder_name):
    
    command_lines = ""
    
    for coordinates,name in zip(coordinates_lists,name_list):
        command_lines += insert_curve_file(coordinates,name)
    
    command_lines += insert_feature_tree_folder(name_list, folder_name)
    
    return command_lines


if __name__ == "__main__":
    CURVE_COUNTER = 0
    macro_contents = set_preamble('MacroSW') + insert_curve_file([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]],"Section 1 Wing") + set_end_code()
    
    # Open a new file for writing
    file = open("newfile1.bas", "w")
    
    # Write to the file
    file.write(macro_contents)
    
    # Close the file
    file.close()