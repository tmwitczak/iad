import os
import shutil
import subprocess

# //////////////////////////////////////////////////////////////////////////// #
print('Preparing data sets:')

# ---------------------------------------------------------------------------- #
print('> Unzipping \'data.zip\'')
subprocess.call('unzip-data.py',
                shell=True)

# ---------------------------------------------------------------------------- #
print('> Converting:')
print('  - \'iris.csv\' (\'iris-train.csv\', \'iris-test.csv\')')
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/iris.csv'
                + ' -c 4'
                + ' -o ./prepared-data/iris.csv'
                + ' -n ./prepared-data/iris-normalised.csv'
                + ' -s ./prepared-data/iris-standardised.csv'
                + ' -t 0.75',
                shell=True)

print('  - \'seeds.csv\'')
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/seeds.csv'
                + ' -c 7'
                + ' -o ./prepared-data/seeds.csv'
                + ' -n ./prepared-data/seeds-normalised.csv'
                + ' -s ./prepared-data/seeds-standardised.csv'
                + ' -t 0.75',
                shell=True)

print('  - \'digits-train.csv\'')
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/digits-train.csv'
                + ' -c 0'
                + ' -o ./prepared-data/digits-train.csv',
                shell=True)

print('  - \'digits-test.csv\'')
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/digits-test.csv'
                + ' -c 0'
                + ' -o ./prepared-data/digits-test.csv',
                shell=True)

# ---------------------------------------------------------------------------- #
print('> Deleting \'original-data\'')
shutil.rmtree('./original-data')

# ---------------------------------------------------------------------------- #
print('> Renaming \'prepared-data\' to \'data\'')
os.rename('./prepared-data', './data')

# //////////////////////////////////////////////////////////////////////////// #
