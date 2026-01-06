def factorial():
    prod=1
    num=int(input("Enter a number:"))
    for i in range (num,0,-1):
        prod=prod*i
    print(prod)    
factorial()    
