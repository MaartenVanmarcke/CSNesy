from nesy.term import Term, Clause, Variable, Fact
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import product
from nesy.unifier import Unifier
from nesy.substituer import Substituer
from nesy.tree import FactNode, NeuralNode, ANDNode, ORNode, LeafNode, AndOrTree

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


class ForwardChaining(LogicEngine):

    def __init__(self,caching_used=False) -> None:
        self.unifier = Unifier()
        self.substituer = Substituer()
        self.caching_used = caching_used
        self.cache_of_trees = dict()
        super().__init__()

    def flush_cache(self):
        self.cache_of_trees=dict()
    """
    This was the original function:
    def reason(self, program: tuple[Clause], queries: list[Term]):
        # TODO: Implement this

        # Dummy example:

        query = parse_term("addition(tensor(images,0), tensor(images,1), 0)")


        Or = lambda x:  None
        And = lambda x: None
        Leaf = lambda x: None
        and_or_tree = Or([
            And([
                Leaf(parse_term("digit(tensor(images,0), 0)")),
                Leaf(parse_term("digit(tensor(images,1), 0)")),
            ])
        ])

        return and_or_tree"""
    
    def _remove_duplicates(self, lst):
        """ Remove duplicates in a given list. """
        res = []
        for x in lst:
            if x not in res:
                res.append(x)
        return res
    
    def reason(self, program: tuple[Clause], queries: list[Term]):
        """
        Return a list of and-or-trees for each different query in the given queries list.
        """
        queries = self._remove_duplicates(queries)
        queries.sort()
        return self.fol_fc_ask_toAndOrTree(program, queries)
    
    def _fol_fc_ask(self, program: tuple[Clause], query: Term):
        """
        This method checks if the given query is logically entailed by the program.
        The used lgorithm is the forward chaining algorithm as described in the book.
        Basically, we start with all atomic sentences.
        In each iteration, we loop over all clauses in the program, adding the head of the clauses for which all terms in the body are known.
        If we have not found new atomic sentences after looping over all rules in the program,
        we stop the algorithm.
        If the query is found as an atomic sentence, we return True,
        else we return False. 

        Arguments
        ---------
            program: A list of clauses
            query: The term to check
            
        Returns
        -------
            True if there exists a derivation for the query; False otherwise.
        """

        self.var_count = 0 # A counter, used to make unused Variables.
        KB = list(program) 
        # Extract all atomic sentences in the KB and remove them from the KB
        atomicSentences = self._extractAtomicSentences_original(KB) 
        new = [None]


        # Repeat until new is empty
        while len(new)>0:
            
            # new <- {}
            new = []
            # For each rule in KB do
            for rule in KB: # Remark: I do not look at atomic sentences
                
                # (p1 ^ p2 ^ ... ^ pn => q) <- Standardize-Variables(rule)
                updatedRule = self._standardize_variables(rule, {})

                # For each theta such that Subst(theta, p1 ^ p2 ^ ... ^ pn) = Subst(theta, p1' ^ p2' ^ ... ^ pn') for some p1', p2', ..., pn' in KB
                for ps in list(product(atomicSentences, repeat=len(updatedRule.body))):
                    subst = self.unifier.unifyMultiple(updatedRule.body, list(ps)) # Remark: I take the most general theta.
                    
                    if subst != None:
                        # q' <- Subst(theta, q)
                        newAtom = self.substituer.substitution(subst, updatedRule.head)

                        # If q' does not unify with some sentence already in KB or new
                        if all([self.unifier.unify(newAtom, sentence) == None for sentence in (atomicSentences + new)]):
                            # Add q' to new
                            new.append(newAtom)
                            # phi <- Unify(q', query)
                            # If phi is not fail, then return phi
                            phi = self.unifier.unify(newAtom, query)
                            if (phi != None):
                                return self.substituer.substitution(phi, query)
            # add new to KB
            # Remark: I add it to the atomicSentences, s.t. I do not have to loop over them as rules.
            atomicSentences  = atomicSentences+new
        # Return false
        return False
    
    def fol_fc_ask_toAndOrTree(self, program: tuple[Clause], queries: list[Term]):
        """
        This method construct an and-or-tree for the given query.
        The algorithm is based on the forward chaining algorithm as implemented in _fol_fc_ask().
        Basically, we start with all atomic sentences.
        In each iteration, we loop over all clauses in the program, adding the head of the clauses for which all terms in the body are known.
        If we add such a head, we also construct an and-or-tree for that head as an chain of and-nodes containing all the and-or-trees of the terms in the body of the rule.
        If we find a new derivation for that same term, we construct an and-or-tree as an or-node. 
        If we have not found new atomic sentences after looping over all rules in the program,
        we stop the algorithm and return the and-or-tree of the query.
        If the query has no possible derivation, the result is a Fact Node with weight 0.

        Arguments
        ---------
            program: A list of clauses
            query: The term to make an and-or-tree for
            
        Returns
        -------
            The and-or-tree of the query. If the query has no possible derivation, the result is a Fact Node with weight 0.
        """
        # define the list of results
        res = [None] * len(queries)
        # check if already cached:

        if self.caching_used:
            for q in range(len(queries)):
                cached_tree= self.cache_of_trees.get(str(queries[q]))
                if cached_tree is not None:
                    # print("Cache found!")
                    res[q] = cached_tree
                else:
                    print("no cache")
            #check if every query was cached -> result can already be returned!        
            if None not in res:
                return AndOrTree(res, queries)


        self.var_count = 0  # A counter, used to make unused Variables.
        KB = list(program)
        # Extract all atomic sentences in the KB and remove them from the KB
        atomicSentences = self._extractAtomicSentences(KB)
        ## New: now, atomicSentences and new are dictionaries. This is to easily construct the and-or-tree.
        # The keys of this dictionary are the string representations of the atomic sentences. 
        # The values of this dictionary are tuples with the first element equal to the atomic sentence 
        # and the second element equal to its and-or-tree node.
        new = {None: None}
        ## New: keep track of the atomic sentences that were added in the previous iteration,
        # This will be necessary to check if a derivation is a new one or not.
        last = {}

        # Repeat until new is empty
        while len(new.keys())>0:
            
            # new <- {}
            new = {}
            # For each rule in KB do
            for rule in KB: # Remark: I do not look at atomic sentences
                
                # (p1 ^ p2 ^ ... ^ pn => q) <- Standardize-Variables(rule)
                updatedRule = self._standardize_variables(rule, {})

                # For each theta such that Subst(theta, p1 ^ p2 ^ ... ^ pn) = Subst(theta, p1' ^ p2' ^ ... ^ pn') for some p1', p2', ..., pn' in KB
                for ps in list(product(self._aux(atomicSentences), repeat=len(updatedRule.body))): #TODO: his self._aux(atomicSentences) leads to the infinite loop (it is very long)!
                    subst = self.unifier.unifyMultiple(updatedRule.body, list(ps)) # Remark: I take the most general theta.

                    if subst != None:
                        # q' <- Subst(theta, q)
                        newAtom = self.substituer.substitution(subst, updatedRule.head)
                        substitutedrule = self.substituer.substitution(subst, updatedRule)

                        # NEW: Checking whether "q' does not unify with some sentence already in KB or new" is not appropriate anymore,
                        # as we want to find all possible derivations
                        
                        # NEW: If the new atom was already derived, then it is possible that we have a new derivation.
                        if str(newAtom) in atomicSentences.keys():
                            # If one of the terms in the body was derived in the previous iteration, then it is a new derivation (else it would have been derived before).
                            flag = False
                            for b in substitutedrule.body:
                                if str(b) in last.keys():
                                    flag = True
                            if flag:
                                # It is a new derivation, so create an OR Node with the previous and-or-tree as a child and the new derivation as the other child.
                                if isinstance(atomicSentences[str(newAtom)][1], ORNode):
                                    # this is just some renaming to have pretty printing, this is not important to construct the and-or-tree
                                    atomicSentences[str(newAtom)][1].name = ""
                                atomicSentences[str(newAtom)] =(newAtom, ORNode(self._makeAnd(substitutedrule.body, atomicSentences, str(newAtom)), atomicSentences[str(newAtom)][1], str(newAtom)))
                        # NEW:  If the new atom was already derived this iteration, then this is also a new derivation.
                        elif str(newAtom) in new.keys():
                            # Create an OR node in an analogous way.
                            if isinstance(new[str(newAtom)][1], ORNode):
                                # again some pretty printing.
                                new[str(newAtom)][1].name = ""
                            new[str(newAtom)] =(newAtom, ORNode(self._makeAnd(substitutedrule.body, atomicSentences, str(newAtom)), new[str(newAtom)][1], str(newAtom)))
                        # The new atom has not been derived yet, so just create an AND node.
                        else:
                            # Add q' to new
                            new[str(newAtom)] = (newAtom, self._makeAnd(substitutedrule.body, atomicSentences, str(newAtom)))
                        
                        # NEW: Checking if a valid unification between q' and query exists, is not appropriate anymore
                        # as we want to find all possible derivations

            # add new to KB
            # Remark: I add it to the atomicSentences, s.t. I do not have to loop over them as rules.
            last = new
            atomicSentences.update(new)

        # Return the constructed tree for each query in queries.
        # If not tree has been constructed for a query, it means it has no derivation and thus is always false.
            # So that is equivalent to a fact node of weight 0.

        for query_index in range(len(queries)):
            query = queries[query_index]
            if res[query_index] is None:
                if str(query) in atomicSentences.keys():
                    tree_query = atomicSentences[str(query)][1]
                    res[query_index] = tree_query 
                    # cache this tree for the future!
                    self.cache_of_trees[str(query)]= tree_query
                else:
                    res[query_index] = FactNode(0, name = str(query)) 
                
        return AndOrTree(res, queries)
    
    def _standardize_variables(self,rule: Clause|Term|Variable, subst: dict[str, Variable] = {}):
        """
        Return a rule equal to the given rule, but its variables replaced by new, unused variables.
        This method needs to be called in which the 2nd parameter equals a new, empty dictionary!
        This 2nd parameter is used as an accumulator.
        """
        # If the rule is a variable, then create a new variable and add this to the subst dictionary.
        if isinstance(rule, Variable):
            # If a new variable has already been made, then use that one, else errors could occur.
            if rule.name not in subst.keys():
                subst[rule.name] = self._newVar()
            return subst[rule.name]
        # If the rule is a term, then create a new term in which all arguments have been updated with new variables.
        if isinstance(rule, Term):
            newargs = [self._standardize_variables(arg, subst) for arg in rule.arguments]
            return Term(rule.functor, tuple(newargs))
        # If the rule is a clause, then create a new clause in which the head and body has been updated with new variables.
        if isinstance(rule, Clause):
            head = self._standardize_variables(rule.head, subst)
            body = [self._standardize_variables(b, subst) for b in rule.body]
            return Clause(head, body)
        else:
            return rule

    def _newVar(self):
        """
        Create a new, unused variable.
        """
        self.var_count += 1
        return Variable("X"+str(self.var_count))

    def _extractAtomicSentences_original(self, KB: list[Clause]) -> list[Term]:
        """
        Scan the knowledge base to find the atomic sentences.
        Atomic sentences are clauses with an empty body.
        These atomic sentences are removed from the given knowledge base.

        Arguments
        ---------
            KB: A list of clauses

        Returns
        -------
            A list with the heads of these clauses with an empty body.
        """
        res = []
        for rule in KB:
            if len(rule.body) == 0:
                res.append(rule)
        for rule in res:
            KB.remove(rule)
        return [rule.head for rule in res]
        
    def _extractAtomicSentences(self, KB: list[Clause]) -> dict[str, tuple[Term, LeafNode]]:
        """
        Scan the knowledge base to find the atomic sentences.
        Atomic sentences are facts (with a weight or a neural network) and clauses with an empty body.
        These atomic sentences are removed from the given knowledge base.

        Arguments
        ---------
            KB: A list of clauses

        Returns
        -------
            A dictionary. The keys of this dictionary are the string representations of the atomic sentences.
            The values of this dictionary are tuples with the first element equal to the atomic sentence and the
            second element equal to its and-or-tree node.
        """
        atomicSentences = []
        res = {}
        for rule in KB:
            if isinstance(rule, Fact):
                atomicSentences.append(rule)
                if rule.weight == None:
                    res[str(rule.term)] = (rule.term, FactNode(1, True, str(rule.term)))
                elif rule.weight.functor == "nn":
                    # TODO: the model is just a string right now, we need to pass this later on!!!
                    res[str(rule.term)] = (rule.term, NeuralNode(rule.weight.arguments[0], rule.weight.arguments[1].arguments[1], rule.weight.arguments[2], name=str(rule.term)))
                else:
                    res[str(rule.term)] = (rule.term, FactNode(float(str(rule.weight)), name = str(rule.term)))

                

            if isinstance(rule, Clause) and len(rule.body) == 0:
                atomicSentences.append(rule)
                # A clause with an empty body, is just a fact with weight 1 as it is always satisfied.
                res[str(rule.head)] = (rule.head, FactNode(1, True, str(rule.head)))
        for rule in atomicSentences:
            KB.remove(rule)
        return res
        
    def _aux(self, d):
        "An auxilary function that returns the list of first elements of the values in a given dictionary."
        return [x[0] for x in d.values()]
    
    def _makeAnd(self, lst: list[Term], atomicSentences: dict[str, tuple[Term, LeafNode]], name: str =""):
        """
        Make a big AND structure as a chain of AND nodes, if a clause has more than 2 terms in its body.
        """
        # If you want to make an AND structure of no terms, then it means we have a clause with an empty body.
        # This means it is just a fact with weight 1 as it is always satisfied.
        # Notice that this will never occur due to the method self._extractAtomicSentences().
        if len(lst) == 0:
            return FactNode(1, name = name)
        # If you want to make an AND structure of just one term, then it means we have a clause with one term in its body.
        # This can be seen as an a clause with two terms in its body in which the second term is always satisfied (True).
        # Hence, we can write this as an AND Node with one child that has weight 1.
        if len(lst) == 1:
            return ANDNode(FactNode(1, True, "True"), atomicSentences[str(lst[0])][1], name= name) 
        if len(lst) == 2:
            return ANDNode( atomicSentences[str(lst[0])][1],  atomicSentences[str(lst[1])][1], name = name)
        # If you want to make an AND structure of more than two terms, we can make it as a chain of AND nodes with two children each.
        return ANDNode(atomicSentences[str(lst[0])][1], self._makeAnd(lst[1:], atomicSentences),  name=name)