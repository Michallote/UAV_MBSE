# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 07:06:21 2023

@author: Michel Gordillo
"""

import xml.etree.ElementTree as ET


def xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    def parse_element(element):
        
        attributes_names = [name.tag for name in element]
        repeated_attributes = set([name for name in attributes_names if attributes_names.count(name) > 1])
        attributes = {attribute : [] for attribute in repeated_attributes}
        
        for child in element:
            if len(child) == 0: #Element has no further ramifications
                #print(child.tag + ' '+ str(len(child)))
                attributes[child.tag] = parse_type(child.text)
                #print(interpret_format(child.text))
                
            else: #Element has ramifications
                if child.tag in repeated_attributes: #Different instance of same type of branch (i.e Main Wing & Fin)
                    attributes[child.tag].append(parse_element(child))
                else: #Unique attribute
                    attributes[child.tag] = parse_element(child)
     
        return attributes
    return parse_element(root)    


def is_float(text):
    return text.replace(' ','').replace('.','').isnumeric()
    

def parse_type(text):
    
    if isinstance(text, str):
        
        if is_float(text):
            return float(text)
        
        if ',' in text:
            vector = text.split(',')
            if all([is_float(n) for n in vector]):
                return [float(n) for n in vector]
        
    return text
        
        

        
        

XMLfile = 'E:/Documentos/Thesis - Master/Diagrams/Glider.xml'


dictionary2 = xml_to_dict(XMLfile)