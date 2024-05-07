import camelot

# PDF file to extract tables from
file = "/home/ubuntu/covid19_ie_removed.pdf"
tables = camelot.read_pdf(file)
# number of tables extracted
print("Total tables extracted:", tables.n)
print(tables[0].df)