import numpy as np

a = np.array([
    ([1346,1265,1170,1030,953,850,794,728]), ([1047,979,896,776,708,620,570,515])
])

b = np.array([
    ([1291,1146,1008,857,757,659]),([1044,917,797,661,573,487])
])

c = np.array([
    ([1398,1320,1233,1147,993,929,822,747]),([1047,985,911,838,709,655,566,503])
])

d = np.array([
    ([1451,1360,1272,1180,1103,976,715,856,785]),([1047,976,903,828,768,662,612,565,506])
])



a_fit = np.polyfit(a[1],a[0],1)
b_fit = np.polyfit(b[1],b[0],1)
c_fit = np.polyfit(c[1],c[0],1)
d_fit = np.polyfit(c[1],c[0],1)
print(a[0])