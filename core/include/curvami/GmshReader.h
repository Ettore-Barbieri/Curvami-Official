#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "curvami/MeshfreeData.h"

namespace GmshReader {

    // A simple struct to hold the mesh connectivity
    struct Element {
        int type;
        int entity_tag; // NEW: Stores the Gmsh Curve/Surface ID
        std::vector<int> node_indices;
    };

    // Helper function to determine node count from Gmsh element types
    inline int getNumNodesForElementType(int elementType) {
        switch (elementType) {
            case 15: return 1; // Point
            case 1:  return 2; // Line
            case 2:  return 3; // Triangle
            case 3:  return 4; // Quadrangle
            default:
                std::cerr << "Warning: Unsupported element type " << elementType << std::endl;
                return 0;
        }
    }

    inline bool readMesh(const std::string& filename, std::vector<Node>& nodes, std::vector<Element>& elements) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open Gmsh file: " << filename << std::endl;
            return false;
        }

        std::string line;
        std::unordered_map<int, int> tag_to_index; // The Hash Table

        while (std::getline(file, line)) {
            
            // ==========================================
            // PARSE NODES BLOCK
            // ==========================================
            if (line.find("$Nodes") != std::string::npos) {
                size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;
                file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;

                nodes.reserve(numNodes);
                int current_index = 0;

                for (size_t b = 0; b < numEntityBlocks; ++b) {
                    int entityDim, entityTag, parametric, numNodesInBlock;
                    file >> entityDim >> entityTag >> parametric >> numNodesInBlock;

                    // Read the tags for this block
                    std::vector<int> tags(numNodesInBlock);
                    for (int i = 0; i < numNodesInBlock; ++i) {
                        file >> tags[i];
                    }

                    // Read the coordinates for this block
                    for (int i = 0; i < numNodesInBlock; ++i) {
                        double x, y, z;
                        file >> x >> y >> z;

                        Node n;
                        n.id = current_index;
                        // Map the 3D Gmsh coordinate to our 2D reference plane
                        n.X_ref = Eigen::Vector2d(x, y);
                        n.u = Eigen::Vector3d::Zero();
                        
                        nodes.push_back(n);
                        
                        // Populate the translation dictionary
                        tag_to_index[tags[i]] = current_index;
                        current_index++;
                    }
                }
            }
            
            // ==========================================
            // PARSE ELEMENTS BLOCK
            // ==========================================
            else if (line.find("$Elements") != std::string::npos) {
                size_t numEntityBlocks, numElements, minElementTag, maxElementTag;
                file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;

                for (size_t b = 0; b < numEntityBlocks; ++b) {
                    int entityDim, entityTag, elementType, numElementsInBlock;
                    file >> entityDim >> entityTag >> elementType >> numElementsInBlock;

                    int nodes_in_elem = getNumNodesForElementType(elementType);

                    // With an unknown type we do not know how many node tags follow,
                    // so skipping the block would desynchronise the stream and silently
                    // corrupt every element after it. Refuse the mesh instead.
                    if (nodes_in_elem == 0) {
                        std::cerr << "Error: unsupported Gmsh element type " << elementType
                                  << " in " << filename
                                  << "; cannot determine its node count, aborting mesh read."
                                  << std::endl;
                        return false;
                    }

                    for (int i = 0; i < numElementsInBlock; ++i) {
                        int elementTag;
                        file >> elementTag;
                        
                        Element el;
                        el.type = elementType;
                        el.entity_tag = entityTag;
                        
                        for (int j = 0; j < nodes_in_elem; ++j) {
                            int nodeTag;
                            file >> nodeTag;
                            // Translate the Gmsh tag to our C++ index. operator[] would
                            // default-insert 0 for an unknown tag and quietly attach the
                            // element to node 0, so look it up explicitly.
                            auto it = tag_to_index.find(nodeTag);
                            if (it == tag_to_index.end()) {
                                std::cerr << "Error: element references undefined node tag "
                                          << nodeTag << " in " << filename << std::endl;
                                return false;
                            }
                            el.node_indices.push_back(it->second);
                        }

                        elements.push_back(el);
                    }
                }
            }
        }
        
        if (nodes.empty() || elements.empty()) {
            std::cerr << "Error: " << filename << " yielded " << nodes.size() << " nodes and "
                      << elements.size() << " elements; the file is empty or malformed."
                      << std::endl;
            return false;
        }

        std::cout << "Successfully loaded mesh: " << nodes.size() << " nodes, " << elements.size() << " elements." << std::endl;
        return true;
    }
}
