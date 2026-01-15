names=("John","Adam","Alice")
marks=(34,56,23)
data=dict(zip(names,marks))
ask=input("Enter the student's name:")
if ask in data:
  print(f"{ask}'s marks is {data[ask]}:")
else:
  print("Student  not found")  