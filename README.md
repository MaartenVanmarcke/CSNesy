# CSNesy

Capita selecta: Assignment
Forward chaining: start with the simple; improve if time left
TODO:
-	Gitignore
-	Inits

Deze week: code begrijpen; nadenken over and or boom structuur
Volgende week: dinsdag : structuur bekijken (5/12)
12/12: evaluator (Eli), logic (Maarten) en backwards (Eli)
17/12 : integreren  & experimenteren
22/12 : experimenteren & report


## Questions:

- isnt it redundant to give queries to evaluator since they are IN the tree(s)?
- How to handle NN: give already to tree (makes sense) or not (as suggestedby the skeleton)?
- NN models are now still strings -> need to change! Maybe give the evaluator a dict with the names of NN (the strings already in the node) + the real NN?

- Why seperation of "images" and tensor images?   Why is the tensor only passed when evaluating? Maybe makes more sense to insert them in tree directly? OR now andor tree just build from strings/queries?

## Further optimizations:

- maybe optimized Forward Chaining (but only maybe)
- use an application where more than 1 NN is needed -> could be interesting


# Used:

$ source /Users/eli/Documents/unif/master_2/capita_selecta/CSNesy/.venv/bin/activate

$ tensorboard --logdir logs/
-> ga naar http://localhost:6006 voor training loss 


[add(0,0,0), add(0,1,1), add(1,0,1), add(1,1,2), digit(tensor(images,0),0), digit(tensor(images,0),1), digit(tensor(images,1),0), digit(tensor(images,1),1)]

