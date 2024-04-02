import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))
sys.path.insert(0,os.path.join(current, 'src', 'nesy'))
from nesy.parser import parse_program, parse_clause
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from itertools import product
from torch.utils.data import default_collate

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])

class Gener:
    def __init__(self, num_digits):
        self.num_digits = num_digits
        self.lst = self.generateList()
        self.lst.pop(0)

    def generateList(self):
        return self._generateList(0, [[]])

    def _generateList(self, curr, ax):
        if curr == self.num_digits:
            return ax
        lst = []
        for i in ax:
            lst.append([0]+i)
            lst.append([1]+i)
        return self._generateList(curr+1, lst)

    def __iter__(self):
        return self.lst.__iter__()


class AdditionTask(Dataset):

    def __init__(self, n=2, train=True, n_classes=10, nr_examples=None):
        self.train = train

        # We iterate over the MNIST dataset to apply the transform
        self.original_images = []
        self.original_targets = []
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)
        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)


        self.n_classes = n_classes # number of possible results of the sum
        self.num_digits = n  # number of digits to sum together

        # GOAL construct the program_string that consists of multiple parts
        # >> Part 1: addition(X_0,...,X_i,Z):-  (-> where i is the n_digits)
        # >> Part 2: digit(X_0,N_0), digit(...,...), digit(X_i,N_i), 
        # >> Part 3: add(N_0,N_1,Z_0), ..., add(N_i,Z_j,Z)  (-> with the intermediate results to always add two terms in the case of n_digits>2)
        # >> Part 4: add(0,0,0), add(1,0,1), ....  (-> the n_classes and n_digits determines the maximum value possible)
        # >> Part 5: nn(digit,tensor(images,X),Y)::digit(tensor(images,X),Y)  (-> Y determined by the n_classes and X by the n_digits)

        # STEP 1: 
        program_string = "addition("

        # STEP 2 -> construct Part 1 and Part 2: depends on n_digits
        # >> 2 auxilery variables
        # >>>> the program_string_helper constructs the part of digit(X_i,N_i)
        program_string_helper = ""
        # >>>> The vars keep track of all variables that have to added
        vars = []
        for i in range(self.num_digits):
            program_string +=f"X_{str(i)},"
            program_string_helper += f"digit(X_{i},N_{i}), "
            vars.append(f'N_{i}')
        program_string += "Z):- "
        # >> add the digit-declarations to the program string
        program_string += program_string_helper
        interm_res_i = 0
        
        # STEP 3 -> Part 3
        # >> create intermediate results, (add 2 of the elements in vars) until there are only 2 elements left. Since then they will lead to the final result
        # >> The form if this intermediate result is add(N_0,N_1,Z_0) or add(Z_0,N_2,Z_1)
        while(len(vars)>2):
            program_string += f"add({vars[0]}, {vars[0+1]}, Z_{interm_res_i}), "
            # the first two elements of vars are deleted (since they have just been processed)
            # the intermediate result Z_i is added to var -> it needs to be added as well! 
            vars[1] = f'Z_{interm_res_i}'
            vars = vars[1:]
            interm_res_i+=1

        # >> The final 2 elements lead to the final result of Z.
        program_string += f"add({vars[0]}, {vars[1]},Z).\n"

        # STEP 4-> Part 4: depends on self.num_digits and self.n_classes
        # >> STEP 4.1: first calculate the maximum value a term can consist of before the final addition
        max_value_term = (self.num_digits -1)*(self.n_classes-1) 
        # >> STEP 4.2: construct all the add(_,_,_)
        for x in range(max_value_term+1):
            for y in range(max_value_term+1):
                if x + y < self.num_digits*(self.n_classes-1) + 1: #the values larger than this self.num_digits*(self.n_classes-1) are not possibe
                     program_string += "\n".join([f"add({x}, {y}, {x + y})."] )

        program_string += "\n"
        # STEP 5 -> Part 5
        program_string += "\n".join(
            [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
             product(range(self.num_digits), range(self.n_classes))])
        
        self.program = parse_program(program_string)
        
        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def __getitem__(self, index):
        images = self.original_images[index * self.num_digits: (index + 1) * self.num_digits]
        targets = self.original_targets[index * self.num_digits: (index + 1) * self.num_digits]
        target = int(targets.sum())

        if self.train:
            # In MNIST Addition, training queries for a single pair of images check for a given sum (i.e. the target)
            # Therefore, we have a List[Term], each element of the list correspond to a single pair of images

            # query = parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(target))[0].term

            terms = "addition("
            for i in range(self.num_digits):
                terms += "tensor(images, " +str(i) + "), "

            terms+= "{}).".format(target)
            query = parse_program(terms)[0].term

            tensor_sources = {"images": images}

            return tensor_sources, query, torch.tensor([1.0])
        
        # For testing/validation set
        else:
            # In MNIST Addition, testing queries for a single pair of images check for all possible sums.
            # In this way, we can compute the most probable sum.
            # Therefore, we have a List[List[Term]], each element of the outer list correspond to a single pair of
            # images. Each element of the inner list correspond to a possible sum.
            queries = []
            for z in range(self.num_digits * (self.n_classes - 1) + 1):

                terms = "addition("
                for i in range(self.num_digits):
                    terms += "tensor(images, " +str(i) + "), "
                
                terms+= "{}).".format(z)
                queries.append(parse_program(terms)[0].term) 
                  
            # queries = [parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(z))[0].term
            #            for z in range(self.n_classes * 2 - 1)]
            tensor_sources = {"images": images}

            #instead of only giving target (=sum of all images), also give targets to be able to calculate the accuracy of individual images
            return tensor_sources, queries, [target,targets]

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                          num_workers=num_workers)

    def __len__(self):
        return self.nr_examples
    
class BaseConverter(Dataset):

    def __init__(self, n=2, train=True, n_classes=10, nr_examples=None):
        '''
        n is the number of the digits to be converted to another base
        n_classes are the nb of possible values for one single digit
        '''
        self.train = train

        # We iterate over the MNIST dataset to apply the transform
        self.original_images = []
        self.original_targets = []
        aux_images = []
        aux_targets = []
        base = 0
        # the base is the first digit
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)
        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)


        self.n_classes = n_classes 
        self.num_digits = n  + 1 # because the base is another given image-input

        # GOAL construct the program_string that consists of multiple parts
        # >> Part 1: addition(X_0,...,X_i,Z):-  (-> where i is the n_digits)
        # >> Part 2: digit(X_0,N_0), digit(...,...), digit(X_i,N_i), 
        # >> Part 3: add(N_0,N_1,Z_0), ..., add(N_i,Z_j,Z)  (-> with the intermediate results to always add two terms in the case of n_digits>2)
        # >> Part 4: add(0,0,0), add(1,0,1), ....  (-> the n_classes and n_digits determines the maximum value possible)
        # >> Part 5: nn(digit,tensor(images,X),Y)::digit(tensor(images,X),Y)  (-> Y determined by the n_classes and X by the n_digits)

        # STEP 1: 
        program_string = "baseConverter("

        # STEP 2 -> construct Part 1 and Part 2: depends on n_digits
        # >> 2 auxilery variables
        # >>>> the program_string_helper constructs the part of digit(X_i,N_i)
        program_string_helper = ""
        # >>>> The vars keep track of all variables that have to added
        vars = []
        # the base is the first digit!
        program_string +=f"X_B,"
        program_string_helper += f"digit(X_B,N_B_1), "
        vars.append("N_B")
        for i in range(self.num_digits-2,-1,-1):
            program_string +=f"X_{str(i)},"
            program_string_helper += f"digit(X_{i},N_{i}), "
            vars.append(f'N_{i}')
        dummy = program_string
        program_string += "Z):- "
        # >> add the digit-declarations to the program string
        program_string += program_string_helper

        for i in range(self.num_digits-1):
            program_string += f"gt(N_B_1, N_{i}), "
        # STEP 3 -> Part 3
        # >> create intermediate results, (add 2 of the elements in vars) until there are only 2 elements left. Since then they will lead to the final result
        # >> The form if this intermediate result is add(N_0,N_1,Z_0) or add(Z_0,N_2,Z_1)
        program_string += f"mul(N_1, N_B_1, Z_1), "
    
        if self.num_digits == 3:
            program_string += f"add(N_{0}, Z_1,Z). "
        else:
            program_string += f"add(N_{0}, Z_1,R_1), "

        for  i in range(2,self.num_digits-1):
            program_string += f"mul(N_B_1, N_B_{i-1}, N_B_{i}), "
            program_string += f"mul(N_{i}, N_B_{i},Z_{i}), "
            if i == self.num_digits-2:
                program_string += f"add(R_{i-1}, Z_{i},Z). "
            else:
                 program_string += f"add(R_{i-1}, Z_{i},R_{i}), "



        # STEP 4-> Part 4: depends on self.num_digits and self.n_classes
        # >> STEP 4.1: first calculate the maximum value a term can consist of before the final addition
        max_value_term = (self.n_classes-1)**(self.num_digits-1)-1#+1
        self.max_value_term = max_value_term
        # >> STEP 4.2: construct all the add(_,_,_)
        for x in range(max_value_term+1):
            for y in range(max_value_term+1):
                     #if x + y <= max_value_term:
                        program_string += "\n".join([f"add({x}, {y}, {x + y})."] )

        for lst in Gener(self.num_digits-1):
            program_string += dummy
            program_string += f"{max_value_term+1}):- "
            for j in range(self.num_digits-1):
                program_string += f"digit(X_{j},N_{j}), "
            for j in range(self.num_digits-1):
                if lst[j] > 0:
                    program_string += f"geq(N_{j},N_B), "
                else:
                    program_string += f"gt(N_B,N_{j}), "
            program_string += f"digit(X_B,N_B). "



        # multiply
        # STEP 4-> Part 4: depends on self.num_digits and self.n_classes
        # >> STEP 4.1: first calculate the maximum value a term can consist of before the final addition
        # >> STEP 4.2: construct all the add(_,_,_)
        for x in range(n_classes):
            for y in range((n_classes-1)**(self.num_digits-2)+1):
                        program_string += "\n".join([f"mul({x}, {y}, {x * y})."] )

        for x in range(n_classes):
            for y in range(x):
                        program_string += "\n".join([f"gt({x}, {y})."] )
                        
        for x in range(n_classes):
            for y in range(x+1):
                        program_string += "\n".join([f"geq({x}, {y})."] )
        
        program_string += "\n"
        # STEP 5 -> Part 5
        program_string += "\n".join(
            [f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in
             product(range(self.num_digits), range(self.n_classes))])
        
        self.program = parse_program(program_string)
        
        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def __getitem__(self, index):
        images = self.original_images[index * self.num_digits: (index + 1) * self.num_digits]
        targets = self.original_targets[index * self.num_digits: (index + 1) * self.num_digits]
        # print("targets", targets)
        if targets[0]==1:
            if np.all(np.array(targets[1:]) == 0):
                target = 0
            else:
                target = self.max_value_term + 1

        elif targets[0]==0:
            target = self.max_value_term + 1
        else:
            try:
                target = int("".join([str(i.item()) for i in targets[1:]]), targets[0])
            except Exception:
                target = self.max_value_term + 1

        if self.train:
            # In MNIST Addition, training queries for a single pair of images check for a given sum (i.e. the target)
            # Therefore, we have a List[Term], each element of the list correspond to a single pair of images

            # query = parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(target))[0].term

            terms = "baseConverter("
            for i in range(self.num_digits):
                terms += "tensor(images, " +str(i) + "), "

            terms+= "{}).".format(target)
            query = parse_program(terms)[0].term

            tensor_sources = {"images": images}

            return tensor_sources, query, torch.tensor([1.0])
        
        # For testing/validation set
        else:
            # In MNIST Addition, testing queries for a single pair of images check for all possible sums.
            # In this way, we can compute the most probable sum.
            # Therefore, we have a List[List[Term]], each element of the outer list correspond to a single pair of
            # images. Each element of the inner list correspond to a possible sum.
            queries = []
            for z in range((self.n_classes-1)**(self.num_digits-1)+1):

                terms = "baseConverter("
                for i in range(self.num_digits):
                    terms += "tensor(images, " +str(i) + "), "
                
                terms+= "{}).".format(z)
                queries.append(parse_program(terms)[0].term) 
                  
            # queries = [parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(z))[0].term
            #            for z in range(self.n_classes * 2 - 1)]
            tensor_sources = {"images": images}

            #instead of only giving target (=sum of all images), also give targets to be able to calculate the accuracy of individual images
            return tensor_sources, queries, [target,targets]

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                          num_workers=num_workers)

    def __len__(self):
        return self.nr_examples