import urllib2
import pandas as pd
import numpy as np
import cv2

IMAGES_PER_PERSON = 10

df = pd.read_csv("a.csv", "\t", header=None, names= ['num', 'name', 'num2', 'img', 'rect', 'checksum'])[1:-1]
print df.tail()
unique_names = df["name"].unique()
count = 0
for i in unique_names:
  print i
  count = 0
  a = df[df['name'].str.contains(i)]
  for j, rows in a.iterrows():
    if count == IMAGES_PER_PERSON:
      continue 
    try:
      resp = urllib2.urlopen(rows["img"])
      image = np.asarray(bytearray(resp.read()), dtype="uint8")
      image = cv2.imdecode(image, cv2.IMREAD_COLOR)
      rect = map(lambda x: int(x), rows["rect"].split(","))
      image = image[rect[1]:rect[3], rect[0]:rect[2]]
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      resized = cv2.resize(image, (100, 100))
      cv2.imwrite("images/{}_{}.jpg".format(i, count), resized)
      count +=1
    except urllib2.HTTPError:
      print "not found!"
    except urllib2.URLError:
      print "URL error!"
    except cv2.error:
      print "OpenCV error!"
    except Exception:
      print "Other error"

# get the data as a csv
#data = urllib2.urlopen("http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt").read()
#data = data.split("\n")[2:]
#splits = map(lambda x:  x.split("\t"), data)
#print splits[:3] 
#df = pd.DataFrame(data = splits) 
#df.to_csv("a.csv", "\t")

