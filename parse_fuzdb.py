import xml.etree.ElementTree as ET
import os
import sys

def main():
    """
    Gets the input and output file paths from the user and starts the parsing process.
    """
    input_file = input("Enter input filepath: ")
    output_file = input("Enter output filepath: ")
    get_sequence(input_file, output_file)

def get_sequence(input_file, output_file):
    """
    Parses the FuzDB XML file to extract and combine fuzzy regions.

    Args:
        input_file (str): The path to the input FuzDB XML file.
        output_file (str): The path to the output file where sequences will be saved.
    """
    if os.path.exists(output_file):
        sys.exit("Given output file already exists!")
        
    with open(output_file, "w", newline="", encoding='utf-8') as output:
        # Parse the XML file
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # Iterate through each <fuzdb> entry in the XML
        for fuzdb in root.findall("fuzdb"):
            entry_id = fuzdb.find("entry_id").text
            sequence = fuzdb.find("sequence").text
            name = fuzdb.find("protein_name").text

            # Extract all fuzzy regions and sort them by their start position
            fuzzy_regions = sorted([
                (int(region.find("start").text), int(region.find("end").text)) 
                if region.find("end").text != "null" else 
                (int(region.find("start").text), int(region.find("start").text))
                for region in fuzdb.findall("fuzzy_region")
            ], key=lambda x: x[0])

            # Merge overlapping or adjacent regions
            combined_regions = []
            for start, end in fuzzy_regions:
                if combined_regions and start - 1 <= combined_regions[-1][1]:
                    combined_regions[-1] = (combined_regions[-1][0], max(end, combined_regions[-1][1]))
                elif (start, end) not in combined_regions:
                    combined_regions.append((start, end))
            
            # Write the combined regions to the output file
            for start, end in combined_regions:
                region_sequence = sequence[start-1:end]
                if len(region_sequence) > 1:
                    output.write(f">{name}_{entry_id}_{start}-{end}\n")
                    output.write(f"{sequence}\n")

if __name__ == "__main__":
    main()

    