# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:25:09 2021

@author: 81807
"""

import numpy as np
import asyncio
import time

async def f1(i):
    
    p = np.random.randint(1,10)
    await asyncio.sleep(p)
    print(i,p)
    
async def main():
    t = []
    for i in range(10):
        t.append(loop.create_task(f1(i)))
    
    await asyncio.wait(t)
    

if __name__ == '__main__':
        
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        print(e)
        pass