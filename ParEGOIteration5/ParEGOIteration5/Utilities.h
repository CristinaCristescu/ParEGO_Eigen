//
//  Utilities.h
//  ParEGOIteration5
//
//  Created by Bianca Cristina Cristescu on 08/02/15.
//  Copyright (c) 2015 Bianca Cristina Cristescu. All rights reserved.
//

#ifndef __ParEGOIteration5__Utilities__
#define __ParEGOIteration5__Utilities__

#include <stdio.h>



class Utilities
{
public:
    void static mysort(int *idx, double *val, int num);
    void static cwr(int **target, int k, int n);
};

#endif /* defined(__ParEGOIteration5__Utilities__) */

