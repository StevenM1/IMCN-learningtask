import numpy as np
import pandas as pd

class Pseudorandomizer(object):
    
    def __init__(self, data, max_identical_iters={'cue': 4, 'correct_answer': 4}):  
        self.data = data
        self.max_identical_iters = {x: y+1 for x, y in max_identical_iters.items()}
                                    # add 1: if 4 rows is allowed, only give an error after 5 identical rows
    
    def check_trial_rows(self, data, row_n): 
        """
        Returns True if any of the conditions for pseudorandomization are violated for the given rows, 
        False if they are fine.
        """
        
        # First, check for the maximum iterations
        for column, max_iter in self.max_identical_iters.items():
            if row_n - max_iter < 0:
                continue

            # Select rows [max_iter-1 - row_n] we're going to check. Never select any row with index < 0
            row_selection = [x for x in np.arange(row_n, row_n-max_iter, -1)]

            # Next, we check if the selected rows only contain *1* trial type. 
            # If so, this means we have max_iter rows of the same trials, and we need to change something.
            if data.iloc[row_selection][column].nunique() == 1:
                return True

        return False

    def run(self, debug=False):
        """
        Pseudorandomizes: makes sure that it is not possible to have more than x iterations for every type of column, specified in columns.
        """
        # Start by copying from original data, and shuffle
        self.data = self.data.sample(frac=1, 
                                     random_state=np.random.randint(0, 1e7, dtype='int')).reset_index(drop=True) 
        
        if debug:
            outer_while_i = 0
            debug_print_after_i = 100

        good_set = False
        while not good_set:
            if debug:
                outer_while_i += 1

            reshuffle = False  # Assume the dataset does not need reshuffling.
            for row_n in range(0, self.data.shape[0]):

                # Select rows [max_iter-1 - row_n] we're going to check

                # Check if the current row, and the (max_iters-1) rows before, are the same value (number of unique values = 1).
                # If so, then move the current row number to the bottom of the dataframe. However, we need to re-check the same four rows again
                # after moving a row to the bottom: therefore, a while loop is necessary.
                checked_row = False
                n_attempts_at_moving = 0
                
                if debug:
                    inner_while_i = 0
                
                while not checked_row:
                    if debug:
                        inner_while_i += 1
                        if inner_while_i > debug_print_after_i:
                            print('New inner loop started for current row')

                    if self.check_trial_rows(self.data, row_n):
                        if debug and inner_while_i > debug_print_after_i:
                            print('Found too many consecutively identical rows.')

                        # If there are too many consecutively identical rows at the bottom of the dataframe, 
                        # break and start over/shuffle
                        if row_n >= (self.data.shape[0] - self.max_identical_iters[list(self.max_identical_iters.keys())[0]]):
                            if debug and inner_while_i > debug_print_after_i:
                                print('These occurred at row_n %d, which is at the bottom of the DF.' % row_n)

                            checked_row = True
                            reshuffle = True

                        # Too many consecutive identical rows? Move row_n to the bottom, and check again with the new row_n.
                        else:
                            if debug and inner_while_i > debug_print_after_i:
                                print('These occurred at row_n %d. Checking the remainder of the DF.' % row_n)

                            # Check if moving to the bottom even makes sense: if all remaining values are identical, it doesn't.
                            if (self.data.iloc[row_n:][self.max_identical_iters.keys()].nunique().values < 2).any():
                                if debug and inner_while_i > debug_print_after_i:
                                    print('All remaining values are identical. I should stop the for-loop, and start over.')

                                checked_row = True
                                reshuffle = True
                            else:
                                if n_attempts_at_moving < 50:
                                    n_attempts_at_moving += 1

                                    if debug and inner_while_i > debug_print_after_i:
                                        print('Not all remaining values are identical. I should move the final part to the bottom.')

                                    # If not, move the current row to the bottom
                                    row_to_move = self.data.iloc[row_n,:]

                                    # Delete row from df
                                    self.data.drop(row_n, axis=0, inplace=True)

                                    # Append original row to end. Make sure to reset index
                                    self.data = self.data.append(row_to_move).reset_index(drop=True)

                                # If we already tried moving the current row to the bottom for 50 times, let's forget about it and restart
                                else:
                                    checked_row = True
                                    reshuffle = True
                    else:
                        if debug and inner_while_i > debug_print_after_i:
                            print('Checked row, but the row is fine. Next row.')
                        checked_row = True

                if reshuffle:
                    good_set = False
                    break  # out of the for loop

                # Reached the bottom of the dataframe, but no reshuffle call? Then we're set.
                if row_n == self.data.shape[0]-1:
                    good_set = True

            if reshuffle:
                # Shuffle, reset index to ensure trial_data.drop(row_n) rows
                self.data = self.data.sample(frac=1, random_state=np.random.randint(0, 1e7, dtype='int')).reset_index(drop=True)
        
        return self.data
