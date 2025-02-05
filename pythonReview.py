professors = ["greg","kianoosh","richard","debra","charlyne","todd","rita"]

myFavoriteProf = professors[2]
print(myFavoriteProf)

print(professors[-1])
print(professors[2:5]) # accessing indices 2, 3, and 4
print(professors[:3]) # accessing indices 0, 1, and 2
print(professors[4:]) # accessing indices 4 all the way to the end
print(len(professors))
print(professors[:])

for i in professors[:]:
    if len(i) > 4:
        professors.remove(i)

print(professors)

professors.append("jason")
print(professors)
professors.extend(["leo","mustafa"])
print(professors)
professors.remove("todd")
print(professors)
professors[3] = "trevor"
print(professors)
professors.insert(3,"waqas")
print(professors)