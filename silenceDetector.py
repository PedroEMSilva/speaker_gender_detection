from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import hilbert, chirp
import numpy as np
from collections import deque,Counter
from itertools import islice
import matplotlib.pyplot as plt

def MovingAveSilence(mylist,N):
    #calculate moving avg
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    #Round the results
    for i in range (len(moving_aves)):
        moving_aves[i] = 10*round(moving_aves[i],2)

    return moving_aves

def RunningMode(seq,N,M):
    """
    Purpose: Find the mode for the points in a sliding window as it
             is moved from left (beginning of seq) to right (end of seq)
             by one point at a time.
     Inputs:
          seq -- list containing items for which a running mode (in a sliding window) is
                 to be calculated
            N -- length of sequence
            M -- number of items in window (window size) -- must be an integer > 1
     Otputs:
        modes -- list of modes with size M - N + 1
       Note:
         1. The mode is the value that appears most often in a set of data.
         2. In the case of ties it the last of the ties that is taken as the mode (this
            is not by definition).
    """
    # Load deque with first window of seq
    d = deque(seq[0:M])

    modes = [Counter(d).most_common(1)[0][0]]  # contains mode of first window

    # Now slide the window by one point to the right for each new position (each pass through
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in islice(seq,M,N):
        old = d.popleft()                      # pop oldest from left
        d.append(item)                         # push newest in from right
        modes.append(Counter(d).most_common(1)[0][0])
    return modes

def FindSilence(x, plot = False):

    x_small = []
    for i in range (len(x) ):
        if round(i%5) == 0:
            x_small.append(x[i])
    x = x_small
    if plot:
        plt.figure(figsize=(14, 5))
        plt.ylim(-1, 1)
        plt.plot(x, label='Raw signal')
        plt.title('Raw signal')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    #Calculating wave envelop
    analytic_signal = hilbert( x)
    amplitude_envelope = np.abs(analytic_signal)

    #Plot wave envelop
    if plot:
        plt.figure(figsize=(14, 5))
        plt.ylim(-1, 1)
        plt.plot(x, label='Raw signal',alpha = 0.3)
        plt.plot(amplitude_envelope, label='Envelop',c = 'r')
        plt.title('Envelop')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    #Rounding wave envelop
    rounded_amplitude_envelope = []
    for i in range (len(amplitude_envelope)):
        rounded_amplitude_envelope.append(round(amplitude_envelope[i],3))

    #Plot wave rounded envelop
    if plot:
        plt.figure(figsize=(14, 5))
        plt.ylim(-1, 1)
        plt.plot(x, label='Raw signal',alpha = 0.3)
        plt.plot(amplitude_envelope, label='Envelop',alpha = 0.6,c = 'm')
        plt.plot(rounded_amplitude_envelope, label='Rounded envelop')
        plt.title('Rounded envelop')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    #Moving Average
    envelop_ave = MovingAveSilence(rounded_amplitude_envelope, 50)

    #Plot wave averaged envelop
    if plot:
        plt.figure(figsize=(14, 5))
        plt.ylim(-10, 10)
        plt.plot([i * 10 for i in x], label='Raw signal',alpha = 0.3)
        #plt.plot(amplitude_envelope, label='Envelop',alpha = 0.2,c = 'm')
        #plt.plot(rounded_amplitude_envelope, label='Rounded envelop',alpha = 0.2)
        plt.plot(envelop_ave, label='Averaged envelop',c= 'g')
        plt.title('Averaged envelop')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')


    #Sampling envelop to increase performace
    sampled_envelop = []


    for i in range (len(envelop_ave) ):
        if round(i%10) == 0:
            sampled_envelop.append(envelop_ave[i])

    x2 =[]
    for i in range (len(x) ):
        if round(i%10) == 0:
            x2.append(x[i])



    #start_time = time.clock()
    #print("--- %s seconds ---" % (time.clock() - start_time))

    #Finding sequences of numbers > 0
    silencePlaces = []
    stackx = []
    stacky = []
    gstackx = []
    gstacky = []
    index = 0




    #For the entire signal
    for i in range (len (sampled_envelop)):
        #if the index is out of bounds, get out
        if index >= len (sampled_envelop):
            break

        atual = sampled_envelop[index]
        
        #if the signal really small, round it to 0
        if atual <= 1:
            
            atual = 0
            #look for more  equal signals i sequence
            for j in range(index,len(sampled_envelop)):
                #if is equal save the position and value (x,y)
                if sampled_envelop[j] <0.25:
                    sampled_envelop[j] = 0
                if sampled_envelop[j] == atual and j+1 < len(sampled_envelop):
                    stackx.append(j)
                    stacky.append(sampled_envelop[j])
                #else analize the size of the actual stack
                else:

                    if atual == 0   :

                        #saving start and end positions
                        if len(stackx)>0:
                            silencePlaces.append([stackx[0],stackx[len(stackx)-1]])

                        #saving new silence in another stack
                        for g in range ( len(stackx)) :
                            gstackx.append(stackx[g])
                            gstacky.append(stacky[g])
                    #jump the stack already analised
                    index += len (stackx)
                    #empty the stacks
                    stackx = []
                    stacky = []
                    break
        index +=1

    #Plot Final Points
    if plot:
        plt.figure(figsize=(14, 5))
        plt.ylim(-1, 1)
    #         new_gstackx = [i * 10 for i in gstackx]
    #         new_sampled_envelop = []
    #         new_x = []
    #         true_x = []
    #         true_y = []
    #         for i in range (len (new_gstackx)-1):
    #             new_x.append(x[new_gstackx[i]])

    #         #true_x = []
    #         true_y = []

    #         for i in range (len (x)):
    #             if i in new_gstackx:
    #                 true_y.append(x[i])
    #             else:
    #                 true_y.append(0)



        plt.plot(x2, label='Raw signal',alpha = 0.3)
        plt.scatter(gstackx,gstacky, label='Silence',c = 'k',s = 0.01)
      #  plt.plot( true_y, label='Detected Rings',c= 'r')
        plt.title('Silence')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
    return silencePlaces

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def FindSilenceFullAudio(x,chunkSize_sec,plot = False):
    soma= 0
    #defining size of chunks
    splits = chunkSize_sec*8000

    #spliting the signal in chunks
    x = list(chunks(x ,splits))

    #list of all rings secs found in the chunks
    globalSilence = []

    #for all chunks
    for i in range (len(x)):

        #empty local ring list
        silencePlaces = []

        #Get local rings
        (silencePlaces) = FindSilence(x[i],plot = plot)

        #for each ring
        for j in range (len(silencePlaces)):

            #get the start and end pos
            start =  silencePlaces[j][0]
            end =  silencePlaces[j][1]

            #calculate pos to sec
            #print start
            #print end

            start = round((float(start)/160)+i*chunkSize_sec,3)
            end = round((float(end)/160)+i*chunkSize_sec,3)

            #save in global list of rings
            if end-start>0.5:
                soma+=end-start
                globalSilence.append([start,end])
        #globalSilence.append([0,0])
    #print("soma silence secs" + str(soma))
    return globalSilence
