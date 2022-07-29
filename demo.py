from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#height, #weight, #shoesize
X = [[123,34,45], [145,56,45], [156,56,34], [178, 67,36], [156, 56, 40], [134, 34, 30], [135, 40,  25]]

y = ['female', 'male', 'male', 'male', 'female', 'female', 'male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
prediction = clf.predict([[133,41,35]]) 

print(prediction) 

clf1 =  KNeighborsClassifier(3) 
clf2= SVC(kernel="linear", C=0.025)
clf3=  GaussianNB()

clf1 = clf1.fit(X,y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)

prediction1 = clf.predict([[156,54,33]]) 
prediction2 = clf.predict([[120,36,43]]) 
prediction3 = clf.predict([[180,65,35]]) 

print(prediction1)
print(prediction2)
print(prediction3) 
