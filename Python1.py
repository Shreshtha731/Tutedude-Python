
try:
  with open("sample.txt",'rt') as fh:
     count=1
     for line in fh:
       print(f"line{count}:{line}")
       count=count+1
except FileNotFoundError :
    print("Error: The file  was not found")   
