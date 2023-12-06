from nesy.term import Term, Variable, Clause

class Substituer():
    """
    A class that concerns applying substitutions on clauses, terms and variables.
    """

    def __init__(self) -> None:
        pass

    def substitution(self, substitution: list[tuple[Variable, Term|Variable|Clause]], query: Term|Variable|Clause):
        """
        Apply the given substitution on the query.

        Arguments
        ---------
            substitution: A list of tuples in which the first element is a variable and the second element is its substitution, that satisfies the method self._isValidSubstition
            query: A term, clause or variable on which you want to apply the substitution.

        Returns
        -------
            The query on which the substitution has been applied
        """
        if not self._isValidSubstition(substitution):
            raise ValueError("The given substitution is not valid.")
        change = True
        # Perform replacement steps as long as changes occur.
        while change:
            change = False
            for (s, t) in substitution:
                query, flag = self.replace(s,t,query)
                if flag:
                    change = True
        return query
    
    def _isValidSubstition(self, substitution: list[tuple[Variable, Term|Variable|Clause]]):
        """
        Check if the substitutions are all variables and if there are no loops in the substitution,
        e.g. [(X,Y), (Y,X)] is invalid.
        
        """
        for (x,y) in substitution:
            if not isinstance(x, Variable) or ((y,x) in substitution) or x in y:
                return False
            
        return True

    def replace(self, s: Variable, t: Term | Variable, u: Variable | Clause | Term) -> tuple[Variable | Clause | Term, bool]:
        """
        Replace s by t in u. 

        Arguments
        ---------
            s: A variable
            t: A term or variable to replace s with in u
            u: A variable, clause or term

        Returns
        -------
            The first return is u in which s is replaced by t.
            The second return is a boolean that denotes whether u has changed or not.


        """

        # s should be a variable
        if isinstance(s, Variable):

            # If u is a variable, then return t if u equals s and else return u.
            if isinstance(u, Variable):
                if (s==u):
                    return t, True
            
            # If u is a term, then return a term with the same functor as u and the same arguments as u.
            # If one of the arguments of u equals s, then substitute t in that argument.
            if isinstance(u, Term):
                res = False
                newargs = []
                for arg in u.arguments:
                    if s == arg:
                        newargs.append(t)
                        res = True
                    else:
                        newargs.append(arg)
                return Term(u.functor, newargs), res
            
            # If u is a clause, then return a clause with the same head and body as u,
            # but in which s is replaced by t.
            if isinstance(u, Clause):
                newhead, res1 = self.replace(s, t, u.head)
                x = [self.replace(s, t, b) for b in u.body]
                newbody, res2 = [v[0] for v in x], any([v[1] for v in x])
                return Clause(newhead, tuple(newbody)), res1 or res2

        else:
            raise ValueError("s is not a Variable.")

        # If the previous cases do not apply, just return u.
        return u, False