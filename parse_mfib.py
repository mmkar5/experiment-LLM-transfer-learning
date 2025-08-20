
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
    Parses the MFIB XML file to extract all protein sequences.

    Args:
        input_file (str): The path to the input MFIB XML file.
        output_file (str): The path to the output file where sequences will be saved.
    """
    if os.path.exists(output_file):
        sys.exit("Given output file already exists!")
        
    with open(output_file, "w", newline="", encoding='utf-8') as output:
        # Parse the XML file
        tree = ET.parse(input_file)
        root = tree.getroot()

        # Iterate through each <entry> in the XML
        for entry in root.findall("entry"):
            accessions = entry.find("accession").text

            # Find all chains within the entry
            for chain in entry.findall("macromolecules/chain"):
                sequence = chain.find("sequence").text
                name = chain.find("name").text
                chain_id = chain.find("id").text
                
                # Write the header and sequence in FASTA-like format
                if sequence:
                    output.write(f">{name}_{accessions}_{chain_id}\n")
                    output.write(f"{sequence}\n")

if __name__ == "__main__":
    main()

    