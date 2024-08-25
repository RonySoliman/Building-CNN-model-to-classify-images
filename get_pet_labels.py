#TODO: 2: Creating Pet Image Labels

# get_pet_labels.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir
import glob
import os #importing it again :)

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """

    results_dic = dict()
    filename_list = listdir(image_dir)
    pet_label = []
    ## Determines number of items in dictionary
    #items_in_dic = len(results_dic)
    #print("\nEmpty Dictionary results_dic - n items=", items_in_dic)

    # Collect all the file names into a list
    #image_names = glob.glob('pet_images/*.jpg')

    for index in range(0, len(filename_list), 1):
        if filename_list[index][0] != ".":
            pet_label = ''
            pet_image_filename = filename_list[index]
            word_list_pet_image_filename = pet_image_filename.lower().split('_')
            pet_name = ''
            
            for word in word_list_pet_image_filename:
                if word.isalpha():
                    pet_name += word + " "
            
            pet_name = pet_name.strip()
            print(f"filename = {pet_image_filename} label = {pet_name}")

            print('\n{:2d} file: {:>25}'.format(index + 1, filename_list[index]))
   
            # count length of empty dictionary
            number_of_items_empty_dic = len(results_dic)
    
            print(f"It's an empty dic {number_of_items_empty_dic}")
    
            # add every element of pet_labels as a value to the results_dic
            if filename_list[index] not in results_dic:
                results_dic[filename_list[index]] = [pet_name]
            else:
                print(f"filename {filenames[index]} already exist {results_dic[filenames[index]]}")
          
    
    # Store the filename and pet_name in the dictionary
    for key in results_dic:
        print(f"filename: {key} pet label: {results_dic[key][0]}")
    
    
    # Replace None with the results_dic dictionary that you created with this
    # function
    return results_dic
