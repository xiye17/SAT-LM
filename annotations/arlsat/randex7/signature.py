### write python code to answer the question
"""
On Tuesday Vladimir and Wendy each eat exactly four separate meals: breakfast, lunch, dinner, and a snack. The following is all that is known about what they eat during that day: At no meal does Vladimir eat the same kind of food as Wendy. Neither of them eats the same kind of food more than once during the day. For breakfast, each eats exactly one of the following: hot cakes, poached eggs, or omelet. For lunch, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet. For dinner, each eats exactly one of the following: fish, hot cakes, macaroni, or omelet. For a snack, each eats exactly one of the following: fish or omelet. Wendy eats an omelet for lunch.
Question: Vladimir must eat which one of the following foods?
Choices:
(A) fish
(B) hot cakes
(C) macaroni
(D) omelet
(E) poached eggs
"""
# declare variables
people = EnumSort([Vladimir, Wendy])
meals = EnumSort([breakfast, lunch, dinner, snack])
foods = EnumSort([fish, hot_cakes, macaroni, omelet, poached_eggs])
eats = Function(people, meals, foods)

# Question: Vladimir must eat which one of the following foods?
# we check whether the options must be true
print(check_valid())