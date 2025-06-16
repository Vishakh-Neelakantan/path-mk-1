import osmnx as ox
import networkx as nx
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
import random
import math
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SmartJoggingRouteGenerator:
    """
    A comprehensive route generator for jogging, walking, and cycling
    that creates loop routes or routes returning close to the start point.
    """
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="jogging_route_generator")
        self.graph = None
        self.start_node = None
        self.start_coords = None
        
    def geocode_location(self, location: str) -> Tuple[float, float]:
        """Convert address/location string to coordinates"""
        try:
            location_data = self.geolocator.geocode(location)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            else:
                raise ValueError(f"Could not geocode location: {location}")
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None
    
    def load_street_network(self, center_coords: Tuple[float, float], 
                           distance: float, mode: str = 'walk') -> nx.MultiDiGraph:
        """
        Load street network from OpenStreetMap around the given coordinates
        
        Args:
            center_coords: (latitude, longitude) tuple
            distance: radius in meters to load network
            mode: 'walk', 'bike', or 'drive'
        """
        try:
            # Map mode to OSMnx network type
            network_type_map = {
                'walk': 'walk',
                'jogging': 'walk',
                'cycling': 'bike',
                'bike': 'bike',
                'drive': 'drive'
            }
            
            network_type = network_type_map.get(mode.lower(), 'walk')
            
            # Load network with a buffer around target distance
            buffer_distance = max(distance * 2, 1000)  # At least 1km buffer
            
            print(f"   Loading {network_type} network within {buffer_distance}m radius...")
            
            self.graph = ox.graph_from_point(
                center_coords, 
                dist=buffer_distance, 
                network_type=network_type,
                simplify=True
            )
            
            # Ensure the graph has edge lengths
            if not any('length' in data for _, _, data in self.graph.edges(data=True)):
                print("   Adding edge lengths to graph...")
                self.graph = ox.add_edge_lengths(self.graph)
            
            # Add edge speeds and travel times for better routing
            try:
                self.graph = ox.add_edge_speeds(self.graph)
                self.graph = ox.add_edge_travel_times(self.graph)
            except:
                print("   Note: Could not add speeds/travel times, using lengths only")
            
            self.start_coords = center_coords
            self.start_node = ox.nearest_nodes(self.graph, center_coords[1], center_coords[0])
            
            return self.graph
            
        except Exception as e:
            print(f"Error loading street network: {e}")
            return None
    
    def calculate_route_distance(self, route_nodes: List) -> float:
        """Calculate total distance of a route in meters"""
        total_distance = 0
        for i in range(len(route_nodes) - 1):
            try:
                # Get edge data between consecutive nodes
                edge_data = self.graph[route_nodes[i]][route_nodes[i + 1]]
                
                # Handle multiple edges between nodes (MultiDiGraph)
                if isinstance(edge_data, dict):
                    # Find the edge with length data
                    edge_length = 0
                    for key, data in edge_data.items():
                        if 'length' in data and data['length'] > 0:
                            edge_length = data['length']
                            break
                    
                    # If no length found, calculate from coordinates
                    if edge_length == 0:
                        node1 = self.graph.nodes[route_nodes[i]]
                        node2 = self.graph.nodes[route_nodes[i + 1]]
                        edge_length = geodesic((node1['y'], node1['x']), (node2['y'], node2['x'])).meters
                else:
                    edge_length = edge_data.get('length', 0)
                    if edge_length == 0:
                        node1 = self.graph.nodes[route_nodes[i]]
                        node2 = self.graph.nodes[route_nodes[i + 1]]
                        edge_length = geodesic((node1['y'], node1['x']), (node2['y'], node2['x'])).meters
                
                total_distance += edge_length
                
            except (KeyError, TypeError):
                # Calculate straight-line distance if edge not found
                node1 = self.graph.nodes[route_nodes[i]]
                node2 = self.graph.nodes[route_nodes[i + 1]]
                dist = geodesic((node1['y'], node1['x']), (node2['y'], node2['x'])).meters
                total_distance += dist
                
        return total_distance
    
    def find_nodes_at_distance(self, target_distance: float, tolerance: float = 0.2) -> List:
        """Find nodes approximately at target distance from start"""
        candidates = []
        
        try:
            # Use Dijkstra to find shortest paths from start node
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, self.start_node, weight='length'
            )
            
            min_distance = target_distance * (1 - tolerance)
            max_distance = target_distance * (1 + tolerance)
            
            for node, distance in lengths.items():
                if min_distance <= distance <= max_distance:
                    candidates.append((node, distance))
            
            # Sort by how close to target distance
            candidates.sort(key=lambda x: abs(x[1] - target_distance))
            
            return candidates[:20]  # Return top 20 candidates
            
        except Exception as e:
            print(f"Error finding nodes at distance: {e}")
            return []
    
    def generate_loop_route(self, target_distance: float, mode: str = 'walk') -> Dict:
        """
        Generate a loop route that returns close to the starting point
        
        Args:
            target_distance: target distance in meters
            mode: transportation mode
            
        Returns:
            Dictionary containing route information
        """
        if not self.graph or not self.start_node:
            return {"error": "Network not loaded"}
        
        try:
            print(f"   Finding candidate waypoints...")
            # Find potential end nodes at roughly half the target distance
            half_distance = target_distance / 2
            candidates = self.find_nodes_at_distance(half_distance)
            
            if not candidates:
                print(f"   No suitable waypoints found at {half_distance}m distance")
                return {"error": "No suitable waypoints found for route generation"}
            
            print(f"   Found {len(candidates)} candidate waypoints")
            print(f"   Testing route combinations...")
            
            best_route = None
            best_distance_diff = float('inf')
            best_total_distance = 0
            
            # Try different combinations of waypoints
            for i, (candidate_node, _) in enumerate(candidates[:10]):  # Try top 10 candidates
                try:
                    # Find shortest path to candidate
                    path_to = nx.shortest_path(
                        self.graph, self.start_node, candidate_node, weight='length'
                    )
                    
                    # Find shortest path back
                    path_back = nx.shortest_path(
                        self.graph, candidate_node, self.start_node, weight='length'
                    )
                    
                    # Combine paths (remove duplicate middle node)
                    full_route = path_to + path_back[1:]
                    
                    # Calculate total distance
                    route_distance = self.calculate_route_distance(full_route)
                    
                    if route_distance > 0:  # Only consider valid routes
                        distance_diff = abs(route_distance - target_distance)
                        
                        print(f"     Route {i+1}: {route_distance/1000:.2f}km "
                              f"(target: {target_distance/1000:.2f}km)")
                        
                        if distance_diff < best_distance_diff:
                            best_distance_diff = distance_diff
                            best_route = full_route
                            best_total_distance = route_distance
                            
                except (nx.NetworkXNoPath, KeyError) as e:
                    print(f"     Route {i+1}: No path found")
                    continue
            
            if best_route and best_total_distance > 0:
                accuracy = 1 - (best_distance_diff / target_distance)
                print(f"   Best route: {best_total_distance/1000:.2f}km "
                      f"({accuracy*100:.1f}% accuracy)")
                
                return {
                    "route_nodes": best_route,
                    "distance_meters": best_total_distance,
                    "distance_km": best_total_distance / 1000,
                    "target_distance": target_distance,
                    "accuracy": accuracy
                }
            else:
                return {"error": "Could not generate suitable route - all candidates resulted in zero distance"}
                
        except Exception as e:
            return {"error": f"Route generation failed: {e}"}
    
    def create_interactive_map(self, route_info: Dict, zoom_start: int = 15) -> folium.Map:
        """Create an interactive map with the route displayed"""
        if "error" in route_info:
            return None
        
        # Create base map centered on start location
        route_map = folium.Map(
            location=self.start_coords,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Get route coordinates
        route_coords = []
        for node in route_info["route_nodes"]:
            node_data = self.graph.nodes[node]
            route_coords.append([node_data['y'], node_data['x']])
        
        # Add route line
        folium.PolyLine(
            locations=route_coords,
            color='red',
            weight=4,
            opacity=0.8,
            popup=f"Distance: {route_info['distance_km']:.2f} km"
        ).add_to(route_map)
        
        # Add start/end marker
        folium.Marker(
            location=self.start_coords,
            popup=f"Start/End Point<br>Route Distance: {route_info['distance_km']:.2f} km",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(route_map)
        
        # Add waypoint markers every ~500m
        if len(route_coords) > 10:
            step = max(1, len(route_coords) // 8)
            for i in range(step, len(route_coords) - step, step):
                folium.CircleMarker(
                    location=route_coords[i],
                    radius=3,
                    popup=f"Waypoint {i//step}",
                    color='blue',
                    fill=True
                ).add_to(route_map)
        
        return route_map
    
    def generate_route_from_location(self, location: str, target_distance_km: float, 
                                   mode: str = 'jogging') -> Tuple[Dict, folium.Map]:
        """
        Complete workflow: geocode location, load network, generate route, create map
        
        Args:
            location: address or location string
            target_distance_km: target distance in kilometers
            mode: transportation mode
            
        Returns:
            Tuple of (route_info, interactive_map)
        """
        print(f"🗺️  Geocoding location: {location}")
        coords = self.geocode_location(location)
        if not coords:
            return {"error": "Could not find location"}, None
        
        print(f"📍 Found coordinates: {coords}")
        
        target_distance_m = target_distance_km * 1000
        print(f"🌐 Loading street network for {mode} mode...")
        
        graph = self.load_street_network(coords, target_distance_m, mode)
        if not graph:
            return {"error": "Could not load street network"}, None
        
        print(f"🛣️  Network loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        print(f"🎯 Generating {target_distance_km}km route...")
        route_info = self.generate_loop_route(target_distance_m, mode)
        
        if "error" in route_info:
            return route_info, None
        
        print(f"✅ Route generated: {route_info['distance_km']:.2f}km "
              f"({route_info['accuracy']*100:.1f}% accuracy)")
        
        print("🗺️  Creating interactive map...")
        route_map = self.create_interactive_map(route_info)
        
        return route_info, route_map

# Example usage and demo
def demo_route_generator():
    """Demonstrate the route generator with example locations"""
    generator = SmartJoggingRouteGenerator()
    
    # Example locations to try
    example_locations = [
        ("Central Park, New York", 5.0, "jogging"),
        ("Hyde Park, London", 3.0, "walking"),
        ("Golden Gate Park, San Francisco", 7.0, "cycling")
    ]
    
    print("🏃‍♂️ Smart Jogging Route Generator Demo")
    print("=" * 50)
    
    for location, distance, mode in example_locations:
        print(f"\n📍 Generating {distance}km {mode} route from {location}")
        print("-" * 40)
        
        route_info, route_map = generator.generate_route_from_location(
            location, distance, mode
        )
        
        if route_map:
            # Save the map
            filename = f"route_{location.replace(' ', '_').replace(',', '')}.html"
            route_map.save(filename)
            print(f"💾 Map saved as: {filename}")
            
            # Print route statistics
            print(f"📊 Route Statistics:")
            print(f"   • Target Distance: {distance} km")
            print(f"   • Actual Distance: {route_info['distance_km']:.2f} km")
            print(f"   • Accuracy: {route_info['accuracy']*100:.1f}%")
            print(f"   • Number of waypoints: {len(route_info['route_nodes'])}")
        else:
            print(f"❌ Error: {route_info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Run the demo
    demo_route_generator()
    
    # Interactive usage example
    print("\n" + "="*50)
    print("🔧 Interactive Usage Example:")
    print("="*50)
    
    generator = SmartJoggingRouteGenerator()
    
    # You can customize these parameters
    location = "Times Square, New York"  # Change this to your preferred location
    distance_km = 4.0  # Change this to your target distance
    mode = "jogging"  # Options: 'walking', 'jogging', 'cycling'
    
    print(f"Generating route from: {location}")
    route_info, route_map = generator.generate_route_from_location(
        location, distance_km, mode
    )
    
    if route_map:
        route_map.save("my_jogging_route.html")
        print("✅ Your custom route has been saved as 'my_jogging_route.html'")
        print("Open this file in your web browser to view the interactive map!")
    else:
        print(f"❌ Could not generate route: {route_info.get('error')}")