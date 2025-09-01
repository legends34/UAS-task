import cv2
import numpy as np
#function to load and return image
def load_image(path):
    image=cv2.imread(path)
    if image is None:
        print("could not load image")
    else:
        print("Image succesfully loaded")

        return image
    

#function to detect land and sea
def detect_land_and_sea(img): # to convert into pure rgb 
    # Converted to HSV (better for color detection)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining range for green color in HSV
    lower_green = np.array([1, 144, 7])   # adjust if needed
    upper_green = np.array([85, 255, 255])

    # Creating mask for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Creating a yellow image of same size
    yellow = np.zeros_like(img, np.uint8)
    yellow[:] = (0, 255, 255)   

    # Applying mask
    result = cv2.bitwise_and(yellow, yellow, mask=mask)

    # Combining with original image (where mask is not green keep original)
    final = cv2.addWeighted(result, 1, img, 1, 0)

    # displaying image
    cv2.imshow("Original", img)
    cv2.imshow("Masked Green -> Yellow", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final


#function to detect colour
def detect_colour(img):
    #converted to hsv for better colour detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_ranges = {
    "red": [
        (np.array([0, 120, 70]),  np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])) 
    ],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([45, 100, 100]), np.array([75, 255, 255]))],
    "pink": [(np.array([140, 80, 150]), np.array([170, 255, 255]))],
    "grey": [(np.array([0, 0, 50]), np.array([180, 40, 220]))],
    "blue": [(np.array([90, 80, 150]), np.array([120, 255, 255]))]
    }
    detected_color = None
    pe=0
    # Looping through defined colors
    for color, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            curr_mask = cv2.inRange(hsv, lower, upper)
            if mask is None:
                mask = curr_mask
            else:
                mask = cv2.bitwise_or(mask, curr_mask)
        
        # Counting non-zero pixels in mask
        if cv2.countNonZero(mask) > 0:
            detected_color = color
            if detected_color=="red":
                pe=3 #priority order of emergency
            elif detected_color=="yellow":
                pe=2
            elif detected_color=="green":
                pe=1
            if detected_color=="pink":
                pe=3 #capacity of rescue pad
            elif detected_color=="blue":
                pe=4
            elif detected_color=="grey":
                pe=2
            break
    return pe


#function to detect shapes in the image 
def detect_shapes(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3,5),3)
    _,h = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)


    edged = cv2.Canny(h, 30, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arr=np.empty((0,3))
    dist=np.empty((0,2))
    details=np.empty((0,4))
    casuality=""
    pe=0
    pc=0
    cn=1 #casuality no.

    for c in contours:

        corner = cv2.arcLength(c, True)
        
        approx = cv2.approxPolyDP(c, 0.03 * corner, True)
        
        x= approx.ravel()[0]
        y= approx.ravel()[1]-10

        num_vertices = len(approx)

        if num_vertices == 3:
            casuality= "Elderly"
            pc=2 #priority order of casuality
            pe=detect_colour(img)
            cn+=1

        elif num_vertices == 4:
            casuality= "Adult"
            pc=1
            pe=detect_colour(img)
            cn+=1 
    
        elif num_vertices ==10:
            casuality= "Child"
            pc=3
            pe=detect_colour(img)
            cn+=1
        else:
            casuality="Rescue pad"
            pc=4
            pe=detect_colour(img)

        cv2.drawContours(img, [c], -1, (0, 0, 0), 2)
        cv2.putText(img, casuality, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str(pc*pe), (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, str(cn), (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        arr=np.append(arr,[[cn,pc,pe]],axis=0)
        dist=np.append(dist,[[x,y]],axis=0)
        details=np.concatenate((arr,dist),axis=1)

    return img ,details,cn

    

#function to find distance between the pad and casuality
def pad_casuality_dist(details):

    pad=np.empty((0,4))
    pn=1 #pad no.
    dist_matrix=np.empty((0,5))


    for i in details:
        if i[1]==4:
            pad=np.append(pad,[[pn,i[ 2],i[3],i[4]]],axis=0)
            pn+=1
  
    for i in details:
        if i[1]==4:
            continue
        else:
            for j in pad:
                d=((i[3]-j[2])**2 + (i[4]-j[3])**2)**(1/2)  # found distance between the two points
                dist_matrix=np.append(dist_matrix,[[i[0],i[1],i[2],j[0],d]] ,axis=0) #i[0]=causality no., i[1]=priority of casuality,i[2]=priority of emergency,j[0]=pad no.,d=dist from pad
    return dist_matrix,pad


#function to assign pads to casuality

def pad_casuality(data, pad_capacity, pad_colors):

    # Tracking used pads
    used_pads = {pad: 0 for pad in pad_capacity}

    # Extracting columns
    casualties = data[:,0].astype(int)
    pc = data[:,1]
    pe = data[:,2]
    pads = data[:,3].astype(int)
    distances = data[:,4]

    # Priority (score)
    priority = pc * pe

    # Combining into one array: (casualty, priority, pad, distance, pc, pe)
    matrix = np.column_stack((casualties, priority, pads, distances, pc, pe))
 
    # Sorting by priority(desc) then distance(asc)
    order = np.lexsort((matrix[:,3], -matrix[:,1]))
    matrix = matrix[order]

    # Assignments grouped by pad
    assignments_by_pad = {pad: [] for pad in pad_capacity}

    assigned_cordinates={}

    # Score sums by pad color
    color_sums = {"blue": 0, "pink": 0, "grey": 0}

    # Processing casualties in priority order
    for cas in np.unique(matrix[:,0]):  # each unique casualty
        # Getting all rows for this casualty
        cas_rows = matrix[matrix[:,0] == cas]
        # Sorting by distance
        cas_rows = cas_rows[cas_rows[:,3].argsort()]

        for row in cas_rows:
            pad = int(row[2])
            if pad in pad_capacity and used_pads[pad] < pad_capacity[pad]:
                pc_val, pe_val = int(row[4]), int(row[5])
                score = pc_val * pe_val

                # Adding [pc, pe] to the pad’s assignment list
                assignments_by_pad[pad].append([pc_val, pe_val])
                
            

                # Updating pad usage
                used_pads[pad] += 1

                # Updating color score (map pad → color)
                pad_color = pad_colors.get(pad, None)
                if pad_color in color_sums:
                    color_sums[pad_color] += score

                break  

    # Converting to matrix 
    assignments_matrix = [
        assignments_by_pad[1],  # grey pad
        assignments_by_pad[2],  # pink pad
        assignments_by_pad[3]   # blue pad
    ]

    
    score_sum=([color_sums["blue"],
        color_sums["pink"],
        color_sums["grey"]])

    return color_sums,assignments_matrix, score_sum





# main code 
pad_capacity = {1:2, 2:3, 3:4} #capacity of each pad
    # Pad → color mapping
pad_colors = {
        1: "grey",   # pad1 = grey
        2: "pink",   # pad2 = pink
        3: "blue"    # pad3 = blue
    }
image_rescue_ratio= {} #this dictionary will contain image no. as key and rescue ratio as value
for i in range(1,11): #running this loop to access each image from the file
    image=load_image('images/%d.png'%i) #loaded the image
    #displaying the Original image
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #calling function to mask land as yellow and storing the image as res , the function displays the image as well
    res=detect_land_and_sea(image)
    #calling function to detect shapes
    casuality,details,n =detect_shapes(res) #casuality is an image in which the shapes are labelled as child,adult or elderly, details is an array which contains all the details of casuality and pad (no.,priority order ,coordinates,etc.) and n is the no. of casualities
    #calling function to find distanc between each casuality and each pad
    distance,pad_det=pad_casuality_dist(details) #distance is a matrix conatining details of casuality and its dist from each pad,pad_det contains the details of pad (pad no. ,pad capacity,x coordinate,y coordinate)
   #calling function to assign pad to each casuality
    colour_sum , assignments, score=pad_casuality(distance,pad_capacity,pad_colors)#
    # Printing final matrix
    print("\n Assignments Matrix %d:"%i)
    print(assignments)
    print("\nSummation of scores by pad color of image%d:"%i)
    print(score)


    avg_sum=0
    for j in colour_sum.values():
        avg_sum+=j
    priority_ratio=avg_sum/n
    image_rescue_ratio["image%d"%i]=priority_ratio

sorted_image_rescue_ratio= dict(sorted(image_rescue_ratio.items(), key=lambda item: item[1], reverse=True))
sorted_keys= sorted_image_rescue_ratio.keys()
print("Image sorted by rescue ratio: \n",sorted_keys)



