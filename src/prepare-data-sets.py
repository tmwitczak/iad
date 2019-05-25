import os
import shutil
import subprocess

CONSOLE_WIDTH: int = 80

# //////////////////////////////////////////////////////////////////////////// #
print(CONSOLE_WIDTH * '/')

# ---------------------------------------------------------------------------- #
print('Preparing data sets:')
print()

# ---------------------------------------------------------------------------- #
print('> Deleting \'data\', \'original-data\', \'prepared-data\'')
print()
shutil.rmtree('./data', ignore_errors = True)
shutil.rmtree('./original-data', ignore_errors = True)
shutil.rmtree('./prepared-data', ignore_errors = True)

print('> Unzipping \'data.zip\'')
print()
subprocess.call('unzip-data.py',
                shell = True)

# ---------------------------------------------------------------------------- #
print('> Converting:')
print()
print('  - \'iris.csv\'')
print()
print('      ~ \'iris-train.csv\'')
print('      ~ \'iris-normalised-train.csv\'')
print('      ~ \'iris-standardised-train.csv\'')
print('      ~ \'iris-test.csv\'')
print('      ~ \'iris-normalised-test.csv\'')
print('      ~ \'iris-standardised-test.csv\'')
print()
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/iris.csv'
                + ' -c 4'
                + ' -o ./prepared-data/iris.csv'
                + ' -n ./prepared-data/iris-normalised.csv'
                + ' -s ./prepared-data/iris-standardised.csv'
                + ' -t 0.6',
                shell = True)

print('  - \'seeds.csv\'')
print()
print('      ~ \'seeds-train.csv\'')
print('      ~ \'seeds-normalised-train.csv\'')
print('      ~ \'seeds-standardised-train.csv\'')
print('      ~ \'seeds-test.csv\'')
print('      ~ \'seeds-normalised-test.csv\'')
print('      ~ \'seeds-standardised-test.csv\'')
print()
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/seeds.csv'
                + ' -c 7'
                + ' -o ./prepared-data/seeds.csv'
                + ' -n ./prepared-data/seeds-normalised.csv'
                + ' -s ./prepared-data/seeds-standardised.csv'
                + ' -t 0.6',
                shell = True)

print('  - \'digits-train.csv\'')
print()
print('      ~ \'digits-train.csv\'')
print('      ~ \'digits-normalised-train.csv\'')
print()
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/digits-train.csv'
                + ' -c 0'
                + ' -o ./prepared-data/digits-train.csv'
                + ' -n ./prepared-data/digits-normalised-train.csv'
                + ' -0 0'
                + ' -1 255',
                shell = True)

print('  - \'digits-test.csv\'')
print()
print('      ~ \'digits-test.csv\'')
print('      ~ \'digits-normalised-test.csv\'')
print()
subprocess.call('convert-classification-data.py'
                + ' -i ./original-data/digits-test.csv'
                + ' -c 0'
                + ' -o ./prepared-data/digits-test.csv'
                + ' -n ./prepared-data/digits-normalised-test.csv'
                + ' -0 0'
                + ' -1 255',
                shell = True)

# ---------------------------------------------------------------------------- #
print('> Computing HOG descriptors')
print()
print('  - \'digits-train.csv\'')
print()
print('      ~ \'digits-hog-train.csv\'')
print()
subprocess.call('compute-hog.py'
                + ' -i ./original-data/digits-train.csv'
                + ' -c 0'
                + ' -o ./prepared-data/digits-hog-train.csv'
                + ' -b 9',
                shell = True)

print('  - \'digits-test.csv\'')
print()
print('      ~ \'digits-hog-test.csv\'')
print()
subprocess.call('compute-hog.py'
                + ' -i ./original-data/digits-test.csv'
                + ' -c 0'
                + ' -o ./prepared-data/digits-hog-test.csv'
                + ' -b 9',
                shell = True)

# ---------------------------------------------------------------------------- #
print('> Deleting \'original-data\'')
print()
shutil.rmtree('./original-data', ignore_errors = True)

# ---------------------------------------------------------------------------- #
print('> Renaming \'prepared-data\' to \'data\'')
print()
os.rename('./prepared-data', './data')

# ---------------------------------------------------------------------------- #
print(CONSOLE_WIDTH * '/')
print()

# //////////////////////////////////////////////////////////////////////////// #
