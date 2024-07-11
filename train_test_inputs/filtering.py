import pdb
import numpy

txt = "underground_eigen_test_files_with_gt_under6.txt"
txt2 = "underground_eigen_test_files_with_gt_under6_filtered.txt"

f = open(txt,'r')
f2 = open(txt2,'w')


for i in range(0,3894):
    A = f.readline()
    print(A)
    if (i % 2) == 0:
        f2.write(A)
print(i)
# pdb.set_trace()


f.close()
f2.close()