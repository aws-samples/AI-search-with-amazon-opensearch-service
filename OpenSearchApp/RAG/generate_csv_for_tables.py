import os
import json
import boto3
import io
from io import BytesIO
import sys
from pprint import pprint
from PyPDF2 import PdfWriter, PdfReader
import re
import shutil

file_content = {}
parent_dirname = "/".join((os.path.dirname(__file__)).split("/")[0:-1])
if os.path.isdir(parent_dirname+"/split_pdf"):
    shutil.rmtree(parent_dirname+"/split_pdf")
os.mkdir(parent_dirname+"/split_pdf")

if os.path.isdir(parent_dirname+"/split_pdf_csv"):
    shutil.rmtree(parent_dirname+"/split_pdf_csv")
os.mkdir(parent_dirname+"/split_pdf_csv")


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    #scores = []
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}
                    
                    # get confidence score
                    #scores.append(str(cell['Confidence']))
                        
                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows#, scores


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        if "," in word['Text'] and word['Text'].replace(",", "").isnumeric():
                            text += '"' + word['Text'] + '"' +' '
                        else:
                            text += word['Text'] +' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] =='SELECTED':
                            text +=  'X '
    return text


def split_pages(file_name):
    
    inputpdf = PdfReader(open(file_name, "rb"))
    file_name_short = re.sub('[^A-Za-z0-9]+', '', (file_name.split("/")[-1].split(".")[0]).lower())

    for i in range(len(inputpdf.pages)):
        
        output = PdfWriter()
        output.add_page(inputpdf.pages[i])
        split_file = parent_dirname+"/split_pdf/"+file_name_short+"%s.pdf" % i
        
        with open(split_file, "wb") as outputStream:
            output.write(outputStream)
        table_csv = get_table_csv_results(split_file)
        if(table_csv != "<b> NO Table FOUND </b>"):
            
            output_file = parent_dirname+"/split_pdf_csv/"+file_name_short+"%s.csv" % i
            file_content[output_file] = table_csv

            # replace content
            with open(output_file, "wt") as fout:
                fout.write(table_csv)

            # show the results
            print('CSV OUTPUT FILE: ', output_file)
    return file_content

def get_table_csv_results(file_name):

    with open(file_name, 'rb') as file:
        img_test = file.read()
        bytes_test = bytearray(img_test)
        #print('Image loaded', file_name)

    # process using image bytes
    # get the results
    #session = boto3.Session(profile_name='profile-name')
    client = boto3.client('textract', region_name='us-east-1')
    # {'S3Object': {
    #         'Bucket': 'ml-search-app-access',
    #         'Name': 'covid19_ie_removed.pdf'
    #     }}
    
    response = client.analyze_document(Document={'Bytes': bytes_test}, FeatureTypes=['TABLES'])

    # Get the text blocks
    blocks=response['Blocks']
    #pprint(blocks)

    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        return "<b> NO Table FOUND </b>"

    csv = ''
    for index, table in enumerate(table_blocks):
        csv += generate_table_csv(table, blocks_map, index +1)
        csv += '\n\n'

    
    return csv

def generate_table_csv(table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)

    table_id = 'Table_' + str(table_index)
    
    # get cells.
    csv = ''#Table: {0}\n\n'.format(table_id)
    for row_index, cols in rows.items():
        for col_index, text in cols.items():
            col_indices = len(cols.items())
            csv += text.strip()+"`" #'{}'.format(text) + ","
        csv += '\n'
        
    # csv += '\n\n Confidence Scores % (Table Cell) \n'
    # cols_count = 0
    # for score in scores:
    #     cols_count += 1
    #     csv += score + ","
    #     if cols_count == col_indices:
    #         csv += '\n'
    #         cols_count = 0

    csv += '\n\n\n'
    return csv

def main_(file_name):
    table_csv = split_pages(file_name)
    #print(table_csv)
    return table_csv

    


# if __name__ == "__main__":
#     file_name = "/home/ubuntu/covid19_ie_removed.pdf"
#     main(file_name)
