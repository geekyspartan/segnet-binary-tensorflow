import os
import glob

# dirPath = ""
# dirPath = ""

current_dir = dirPath + "segnet/"

os.chdir(dirPath + "segnet")

list_of_files = glob.glob("10_*.txt")


accuracies = []

print "Number of files " + str(len(list_of_files))


for f in list_of_files:
  f = open(current_dir + f)
  content = float(f.read())
  accuracies.append(content)
print accuracies
print sum(accuracies)/len(accuracies)





