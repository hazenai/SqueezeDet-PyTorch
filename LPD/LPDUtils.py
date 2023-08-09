from shapely import Polygon

def is_lpbox_valid(KPs, LP):
    # KPs = 
    #       Wheel Points        [0=FL, 1=FR, 2=RL, 3=RR]
    #       Roof  Points        [8=FL, 9=FR, 10=RL, 11=RR]
    #       Headlight Points    [4=FL, 5=FR]
    #       Taillight Points    [6=RL, 7=RR]
    #       Wheel Ground Point  [12=FL, 13=FR, 14=RL, 15=RR]
    #       Front Screen Point  [16=LT, 17=RT, 18=LB, 19=RB]
    #       Back Screen Point   [20=LT, 21=RT, 22=LB, 23=RB]
    # LP  = [_numlps, lp, lpscore, lpbox]
    # lpbox = [left, top, bottom, right]
    veh_front_polygon = Polygon([ KPs[5], KPs[4],  KPs[12], KPs[13], KPs[5] ])
    veh_rear_polygon  = Polygon([ KPs[6], KPs[7],  KPs[15], KPs[14], KPs[6] ])
    
    x1, y1, x2, y2 = LP
    lp_polygon = Polygon([ (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1) ])

    if not (veh_front_polygon.is_valid and veh_front_polygon.is_valid and veh_rear_polygon.is_valid):
        return False

    lp_area = lp_polygon.area
    veh_front_inter = veh_front_polygon.intersection(lp_polygon).area / lp_area
    veh_rear_inter = veh_rear_polygon.intersection(lp_polygon).area / lp_area
    
    return veh_front_inter > 0.1 or veh_rear_inter > 0.1