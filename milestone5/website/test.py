from milestone3 import *
zoom=ZoomFeatureClassifier()
X,y=zoom.load()
clf,old,new=zoom.model("linear",X,y)
print(zoom.predict(clf,"new feature"))

from milestone4 import *
webex=WebexFeatureClassifier()
X,y=webex.load()
clf,old,new=webex.model("linear",X,y)
print(webex.predict(clf,"new feature"))