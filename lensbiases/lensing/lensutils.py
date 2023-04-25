import sympy as sp


def get_sin_cos_exprs():
    #https://github.com/simonsobs/symlens/blob/master/symlens/utils.py#L282
    p1, p2 = sp.symbols('p1 p2')
    expr = sp.sin(2*(p1-p2))
    expr = sp.expand_trig(expr)
    l1, l2, l1x, l1y, l2x, l2y = sp.symbols('l1 l2 l1x l1y l2x l2y')

    expr = sp.expand(sp.simplify(expr.subs([(sp.cos(p1),l1x/l1),(sp.cos(p2),l2x/l2),
                                (sp.sin(p1),l1y/l1),(sp.sin(p2),l2y/l2)])))

    lmbdasin = sp.lambdify((l1, l2, l1x, l1y, l2x, l2y), expr, "numpy")

    expr = sp.cos(2*(p1-p2))
    expr = sp.expand_trig(expr)
    expr = sp.expand(sp.simplify(expr.subs([(sp.cos(p1),l1x/l1),(sp.cos(p2),l2x/l2),
                                (sp.sin(p1),l1y/l1),(sp.sin(p2),l2y/l2)])))
    lmbdacos = sp.lambdify((l1, l2, l1x, l1y, l2x, l2y), expr, "numpy")

    return lmbdasin, lmbdacos