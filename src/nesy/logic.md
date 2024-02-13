# Logic
## Substitutions
"A substitution is a finite set of pairs of the form variable /
term, such that all variables at the left-hand sides of the
pairs are distinct.
In our substitutions we will NOT allow that some variable
that occurs left also occurs in some term at the right."

-> dictionary<variable, term>

## Unifiers
Given a set of simple expressions $S$, we call a substitution $\theta$ a unifier for $S$ if:
$S\theta$ is a singleton.

Only the most general substitution $\theta$ allows to derive the strongest conclusions in deduction steps.

I based the unification algorithm on the slides of AI course.

7



Moet de factnode ook de naam bevatten?
Grounding not necessary if no fact has a variable!!!
Heeft een factnode die bool nodig? Want er is geen not aanwezig in onze syntax?