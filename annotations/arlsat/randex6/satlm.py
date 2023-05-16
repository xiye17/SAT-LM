### write python code to answer the question
"""
Workers at a water treatment plant open eight valves—G, H, I, K, L, N, O, and P—to flush out a system of pipes that needs emergency repairs. To maximize safety and efficiency, each valve is opened exactly once, and no two valves are opened at the same time. The valves are opened in accordance with the following conditions: Both K and P are opened before H. O is opened before L but after H. L is opened after G. N is opened before H. I is opened after K.
Question: Each of the following could be the fifth valve opened EXCEPT:
Choices:
(A) H
(B) I
(C) K
(D) N
(E) O
"""
# declare variables
valves = EnumSort([G, H, I, K, L, N, O, P])
opened = Function([valves] -> [int])
ForAll([v:valves], And(1 <= opened(v), opened(v) <= 8))

# constraints
# no two valves are opened at the same time
Distinct([v:valves], opened(v))

# Both K and P are opened before H
And(opened(K) < opened(H), opened(P) < opened(H))

# O is opened before L but after H
And(opened(O) > opened(H), opened(O) < opened(L))

# L is opened after G
opened(L) > opened(G)

# N is opened before H
opened(N) < opened(H)

# I is opened after K
opened(I) > opened(K)

# Each of the following could be the fifth valve opened EXCEPT:
# we check whether the options can possibly be true, and find the exception
# (A)
is_exception(is_sat(opened(H) == 5))
# (B)
is_exception(is_sat(opened(I) == 5))
# (C)
is_exception(is_sat(opened(K) == 5))
# (D)
is_exception(is_sat(opened(N) == 5))
# (E)
is_exception(is_sat(opened(O) == 5))