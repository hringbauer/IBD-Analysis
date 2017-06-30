'''
Created on Feb 4, 2016

@author: Harald
'''


import numpy as np
import simplekml  # To extract Google map files
import matplotlib.pyplot as plt
from geopy.distance import vincenty  # To compute great circle distance from coordinates
from mpl_toolkits.basemap import Basemap


# my_proj = Proj(proj='utm',zone="31T", ellps='WGS84',units='m')   # Prepare the Longitude/Latidude to Easting/Northing transformation
style = simplekml.StyleMap()  # Create a style map for highlight style and normal style FOR UNCORRECTED
style.normalstyle.labelstyle.scale = 0
style.normalstyle.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png"  # grn
style.highlightstyle.labelstyle.scale = 1
style.highlightstyle.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/pushpin/blue-pushpin.png'
######################################################

class LoadData(object):
    '''
    Loads and pre-processes the data. Contains methods for data Cleaning.
    '''
    populations = []  # Contains table of IDs and populations
    blocks = []  # Contains table of all shared blocks
    coordinates = []  # Matrix saving the coordinates of a country
    countries_oi = []
        
    # countries_oi = []  # List of countries of interest
    pw_distances = []  # Pairwise distance Matrix
    pw_blocksharing = []  # Matrix of block sharing between countries
    nr_individuals = []  # List of Nr of Individuals
    position_list = []  # List of all positions; nx2 array of x and y-Values

    def __init__(self, pop_path, ibd_list_path, geo_path, min_block_length, countries_oi):
        '''Runs all the stuff to bring data in shape '''
        self.countries_oi = countries_oi
        print("Loading data...")  

        # First load all the necessary files.
        self.populations = np.loadtxt(pop_path, dtype='string', delimiter=',')[1:, :]  # Load the cross_reference list
        self.blocks = np.loadtxt(ibd_list_path, dtype='string', delimiter=',')[1:, :]
        self.coordinates = np.loadtxt(geo_path, dtype='string', delimiter=',')[1:, :]
        self.populations[:, 1] = [countrie.replace("\"", "") for countrie in self.populations[:, 1]]  # Replace double "" in populations
        
        # self.countries_oi = list(set(self.populations[:, 1]))  # Set countries of interest to everything
        # self.extract_kml(self.coordinates[:, 0], self.coordinates[:, 1].astype('float'), self.coordinates[:, 2].astype('float'))
        
        print(self.countries_oi)
        print("Total number of inds: %.1f" % len(self.populations))
        print("Total number of blocks: %.1f" % len(self.blocks[:, 0]))  
        
        self.calculate_pw_dist()  # Update important fields. Including position_list
        self.calc_ind_nr() 
        k = len(self.countries_oi)
        
        block_list = self.blocks[(self.blocks[:, 3].astype('float') > min_block_length), :]  # Only keep blocks with minimum length
        
        # Replace individual numbers by their country of origin:
        for i in range(len(block_list[:, 0])):
            ind1 = np.where(self.populations[:, 0] == block_list[i, 0])[0][0]
            ind2 = np.where(self.populations[:, 0] == block_list[i, 1])[0][0]
            block_list[i, 0] = self.populations[ind1, 1]
            block_list[i, 1] = self.populations[ind2, 1]
        
        # Generate the block sharing matrix  
        filler = np.frompyfunc(lambda x: list(), 1, 1)
        a = np.empty((k, k), dtype=np.object)
        self.pw_blocksharing = filler(a, a)  # Initialize everything to an empty list.    
        
        for i in range(len(block_list[:, 0])):
            ind1 = np.where(self.countries_oi == block_list[i, 0])[0]  # First locate block1
            ind2 = np.where(self.countries_oi == block_list[i, 1])[0]  # Then the second block            
            
            if not ((len(ind1) > 0) and (len(ind2) > 0)):  # Jump to next iteration if countries not in countrie_oi
                continue
            self.pw_blocksharing[max(ind1[0], ind2[0]), min(ind1[0], ind2[0])].append(float(block_list[i, 3]))  # Add to block sharing.
            # print("Doing something!!!")
           
        print("Interesting block sharing: %.0f " % np.sum([len(a[i, j]) for i in range(k) for j in range(k)]))  # Print interesting block sharing
        

    def calculate_pw_dist(self):
        '''Calculate Pairwise Distances of countries of interest.
        Also save latitude and longitude of every country'''
        country_found_list = []  # List of countries which were found
        lat_list = []
        long_list = []
             
        for country in self.countries_oi:  # For every country of interest
            print(country)
            coord_ind = np.where(self.coordinates[:, 0] == country)[0]
            if not len(coord_ind) > 0:
                print("Country %.s not found!" % country)
                continue
            print("Country %.s has index %i" % (country, coord_ind[0]))
            
            country_found_list.append(country)  # Append Country to the list of found countries
            lat_list.append(self.coordinates[coord_ind, 1])  # Append Geographic coordinates
            long_list.append(self.coordinates[coord_ind, 2])
            
        print(country_found_list)
        self.countries_oi = np.array(country_found_list)  # Only countries which are found are of interest
        
        lat_list = np.array(lat_list).astype('float')
        long_list = np.array(long_list).astype('float')
        
        lat1_list = np.array([i[0] for i in lat_list])  # For kml extraction
        long1_list = np.array([i[0] for i in long_list])
        self.extract_kml(country_found_list, lat1_list, long1_list)
        
        self.make_mpl_map(lat1_list, long1_list)  # Send data to Matplot Lib card function
        
        l = len(self.countries_oi) 
        dist_mat = np.zeros((l, l))
        
        for i in range(0, l):  # Iterate over all pairs of 
            for j in range(0, i): 
                dist_mat[i, j] = self.calc_dist(lat_list[i], long_list[i], lat_list[j], long_list[j])
        
        self.latlon_list = np.array([[lat_list[i], long_list[i]] for i in xrange(l)])  # Write the LatLon List
        self.position_list = self.map_projection(long_list, lat_list)
        
        self.pw_distances = dist_mat
    
        
    def calc_dist(self, lat1, long1, lat2, long2):
        '''Calculates the pairwise distance between the given coordinates''' 
        coord1 = (lat1, long1)
        coord2 = (lat2, long2)
        return vincenty(coord1, coord2).meters / 1000.0  # Return distance of input points (in km)

    
    def calc_ind_nr(self):
        '''Generates the number of individuals per countrie_of_i entry'''
        nr_inds = []
        for country in self.countries_oi:
            nr_ind = np.sum(self.populations[:, 1] == country)
            # print("Country %s has %.0f inds" % (country, nr_ind))
            nr_inds.append(nr_ind)
        self.nr_individuals = nr_inds
                           
    def extract_kml(self, index, lat, lon):
        '''Extract Google maps file from lot long Values with name index'''
        kml = simplekml.Kml()  # Load the KML Creater
    
        for i in range(len(lat)):   
            pnt = kml.newpoint(name=index[i], coords=[(lon[i], lat[i])])   
            # pnt.description = data[i, comm_ind]
            pnt.stylemap = style
            # pnt.altitudemode = 'absolute'
            pnt.extrude = 1
        
        kml.save("output.kml")  # Save the kml file
        print("GPS saved!")
        
        
    def make_mpl_map(self, lats, lons):
        '''Method that makes a map within matplotlib with points at lat, lon'''
        
        m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=65,
                    llcrnrlon=5, urcrnrlon=40, resolution='i')
        # m = Basemap(llcrnrlon=-10.5, llcrnrlat=35, urcrnrlon=4., urcrnrlat=44.,
                    # resolution='i', projection='merc', lat_0 = 39.5, lon_0 = -3.25)
        
        
        # m.drawcountries(linewidth=0.1,color='w')
        
        # m.drawmapboundary(fill_color='aqua')
        m.drawcoastlines()
        m.drawcountries(linewidth=1, color='k')
        # m.drawcountries()
        m.fillcontinents(color='coral')
        
        # Make the points
        for i in range(len(lons)):
            print(lats[i])
            print(lons[i])
            x, y = m(float(lons[i]), float(lats[i]))
            m.plot(x, y, 'bo', markersize=8)
        
        plt.show()
        
    
    def map_projection(self, lon_vec, lat_vec):
        '''
        Winkel projection with standard parallel at mean latitude of the sample
        argument is (n,2) array with longitude as first column and latitude as second column
        returns cartesian coordinates in kilometers
        '''
        lon_lat_positions = np.column_stack((lon_vec, lat_vec))
        earth_radius = 6367.0  # radius at 46N
        lon_lat_positions = np.pi * lon_lat_positions / 180.0  # convert to radian
        mean_lat = np.mean(lon_lat_positions[:, 1])
        X = earth_radius * lon_lat_positions[:, 0] * .5 * (np.cos(mean_lat) + np.cos(lon_lat_positions[:, 1]))
        Y = earth_radius * lon_lat_positions[:, 1]
        return np.column_stack((X, Y))
    
    
