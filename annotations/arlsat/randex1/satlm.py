### write python code to answer the question
"""
Each of five students—Hubert, Lori, Paul, Regina, and Sharon—will visit exactly one of three cities—Montreal, Toronto, or Vancouver—for the month of March, according to the following conditions: Sharon visits a different city than Paul. Hubert visits the same city as Regina. Lori visits Montreal or else Toronto. If Paul visits Vancouver, Hubert visits Vancouver with him. Each student visits one of the cities with at least one of the other four students.
Question: Which one of the following must be true for March?
Choices:
(A) If any of the students visits Montreal, Lori visits Montreal.
(B) If any of the students visits Montreal, exactly two of them do.
(C) If any of the students visits Toronto, exactly three of them do.
(D) If any of the students visits Vancouver, Paul visits Vancouver.
(E) If any of the students visits Vancouver, exactly three of them do.
"""
# declare variables
students = EnumSort([Hubert, Lori, Paul, Regina, Sharon])
cities = EnumSort([Montreal, Toronto, Vancouver])
visits = Function([students] -> [cities])

# constraints
# Sharon visits a different city than Paul
visits(Sharon) != visits(Paul)

# Hubert visits the same city as Regina
visits(Hubert) == visits(Regina)

# Lori visits Montreal or else Toronto
Or(visits(Lori) == Montreal, visits(Lori) == Toronto)

# If Paul visits Vancouver, Hubert visits Vancouver with him
Implies(visits(Paul) == Vancouver, visits(Hubert) == Vancouver)

# Each student visits one of the cities with at least one of the other four students
ForAll([s1:students], Exists([s2:students], And(s2 != s1, visits(s1) == visits(s2))))

# Which one of the following must be true for March?
# we check whether the options must be true
# (A)
is_valid(Implies(Exists([s:students], visits(s) == Montreal), visits(Lori) == Montreal))
# (B)
is_valid(Implies(Exists([s:students], visits(s) == Montreal), Count([s:students], visits(s) == Montreal) == 2))
# (C)
is_valid(Implies(Exists([s:students], visits(s) == Toronto), Count([s:students], visits(s) == Toronto) == 3))
# (D)
is_valid(Implies(Exists([s:students], visits(s) == Vancouver), visits(Paul) == Vancouver))
# (E)
is_valid(Implies(Exists([s:students], visits(s) == Vancouver), Count([s:students], visits(s) == Vancouver) == 3))