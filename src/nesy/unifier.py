from nesy.term import Term, Variable
from nesy.substituer import Substituer

class Unifier():
    """
    A class that concerns unifying terms and variables. (self.unify())
    There is also functionality for unifying two lists of terms and variables. (self.unifyMultiple())
    """

    def __init__(self) -> None:
        self.substituer = Substituer()

    def unifyMultiple(self, s: list[Term|Variable], t: list[Term|Variable]) -> list[tuple[Variable, Term | Variable]]:
        """
        Unify two lists. This means that the most general unifier S is found such that for each element x in s and y in t the following holds: S(x) = S(y)

        Arguments
        ---------
            s: A list of terms and variables.
            t: A list of terms and variables.

        Returns
        -------
            The most general unification between the two lists. If no such mgu is found, return None.

        """

        if len(s) != len(t):
            raise ValueError("The given lists do not have the same length.")
        mgu = [(s[i], t[i]) for i in range(len(s))]
        return self._unifyLoop(mgu)

    def unify(self, s: Term|Variable, t: Term|Variable) -> list[tuple[Variable, Term | Variable]]:
        """
        Unify two terms or variables. This means that the most general unifier S is found such that S(s) = S(t)

        Arguments
        ---------
            s: A term or a variable
            t: A term or a variable

        Returns
        -------
            The most general unification between the two terms or variables. If no such mgu is found, return None.

        """
        
        mgu = [(s, t)]
        return self._unifyLoop(mgu)

    def _unifyLoop(self, mgu: list[tuple[Term | Variable, Term | Variable]]) -> list[tuple[Variable, Term | Variable]]:
        """
        Perform the unification algorithm.

        Arguments
        ---------
            mgu: a list of tuples (Term|Variable, Term|Variable)
            
        Returns
        -------
            The most general unification which is a list of tuples (Variable, Term|Variable). If the algorithm fails, return None.

        """

        change = True
        # Keep trying to update the mgu as long as there has been made a change. 
        # If there is no change, stop.
        while(change):
            change = False
            for i in range(len(mgu)):
                s, t = mgu[i]
                # Unify s with t, and possibly update the tuples in the current mgu.
                # Flag denotes if a change has been made in this step.
                # v is a list with the output of this step.
                v, flag = self._unify(s, t, mgu)
                if flag:
                    change = True

                # If unifying s and t fails, then return None.
                if v == None:
                    return None
                

                k = len(v)
                # If v is an empty list, it means (s,t) has to be deleted from mgu. Set it to None for now.
                if k == 0:
                    mgu[i] = None
                # If v has one element, it means (s,t) has to be changed to this element in mgu
                elif k == 1:
                    mgu[i] = v[0]
                # If v has more than one element, it means that (s,t) has to be deleted and each element in this list has to be added to mgu.
                else:
                    mgu[i] = None
                    for elem in v:
                        mgu.append(elem)


            # Remove all None's in mgu.
            mgu = self._clean(mgu)
        return mgu
        
        
    def _clean(self, mgu: list):
        """
        Remove all occurences of None in a list.
        """
        return [v for v in mgu if v!=None]
    
    def _unify(self, s: Term|Variable, t: Term|Variable, mgu: list):
        """
        Make a case analysis based on whether s and t are terms or variables to know the case in the unification algorithm.

        This returns a tuple in which the first element is the replacement for (s,t) in mgu.
        The second element in this tuple denotes whether mgu has changed.
        """
        if isinstance(t, Variable):
            if isinstance(s, Variable):
                # s and t are variables
                return self._unifyvv(s, t, mgu)
            
            else:
                # s is a term and t is a variable
                return self._unifytv(s, t, mgu)
            
        else:
            if isinstance(s, Variable):
                # s is a variable and t is a term
                return self._unifyvt(s, t, mgu)
                
            else:
                # s is a term and t is a term
                return self._unifytt(s, t, mgu)


    def _unifytv(self, s: Term, t: Variable, mgu: list):
        """
        If s is a term and t is a variable,
        then replace (s, t) by (t, s).
        """
        return [(t, s)], True
    
    def _unifyvv(self, s: Variable, t: Variable, mgu: list):
        """
        If s is a variable and t is a variable and they are the same,
        then delete (s, t) from mgu.
        If they are not the same, then replace all occurrences of s in mgu by t.
        """
        if (s==t):
            return [], True
        else:
            return self._replaceOccurrences(s, t, mgu)


    def _unifyvt(self, s:Variable, t: Term, mgu: list):
        """
        If s is a variable and t is a term and t contains s,
        then the unification algorithm fails and we return None.
        If t does not contain s, then replace all occurrences of s in mgu by t.
        """
        if s in t.arguments:
            return None, True
        else:
            return self._replaceOccurrences(s, t, mgu)
    
    def _unifytt(self, s: Term, t: Term, mgu: list):
        """
        If s is a term and t is a term and the two terms have the same functor and number of arguments,
        then replace (s, t) in mgu by (s1, t1), (s2, t2), ..., (sn, tn) if s = g(s1, s2, ... sn) and t = g(t1, t2, ... tn).
        Otherwise, the unification algorithm fails and we return None.
        """
        if s.functor != t.functor or len(s.arguments) != len(t.arguments):
            return None, True
        res = []
        for i in range(len(s.arguments)):
            res.append((s.arguments[i], t.arguments[i]))
        return res, True
    
    def _replaceOccurrences(self, s:Variable, t:Variable|Term, mgu: list):
        """
        Replace all occurrences of s in mgu by t.
        """
        change = False
        for idx, u in enumerate(mgu):
            if u != None and u[1] != t:
                newu1, flag1 = self.substituer.replace(s, t, u[0])
                newu2, flag2 = self.substituer.replace(s, t, u[1])
                if flag1 or flag2:
                    change = True
                    mgu[idx] = (newu1, newu2)
        return [(s,t)], change