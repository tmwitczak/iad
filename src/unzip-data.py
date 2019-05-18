import zipfile

zip_ref = zipfile.ZipFile('data.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()