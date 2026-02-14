import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model

# Load model
model = create_model('densenet121', pretrained=False, num_classes=38)
model.load_state_dict(torch.load('model/densenet_weights.pth', map_location='cpu'))
model.eval()

# Class names
class_names = [
    'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Apple___Apple_scab',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Disease info database
# Comprehensive Plant Disease Database
plant_disease_database = {
    'Apple___Black_rot': {
        'symptoms': [
            'Dark, circular lesions on leaves with concentric rings',
            'Brown to black cankers on branches and trunk',
            'Fruit develops dark, sunken lesions that eventually cover entire fruit',
            'Infected fruit becomes mummified and shriveled',
            'Leaf spots may have yellow halos'
        ],
        'treatment': [
            'Remove and destroy infected plant material',
            'Apply fungicides containing captan, thiophanate-methyl, or myclobutanil',
            'Prune for good air circulation',
            'Apply dormant oil spray in early spring',
            'Maintain proper sanitation practices'
        ],
        'seasons': ['Spring', 'Summer', 'Fall'],
        'peak_season': 'Summer',
        'prevention': [
            'Plant resistant varieties',
            'Ensure proper spacing for air circulation',
            'Avoid overhead watering',
            'Regular pruning of dead/diseased wood'
        ],
        'severity': 'High',
        'spread_method': 'Fungal spores via wind and rain',
        'optimal_conditions': 'Warm, humid weather (75-85°F)'
    },
    
    'Apple___Cedar_apple_rust': {
        'symptoms': [
            'Bright orange spots on upper leaf surface',
            'Cup-shaped structures (aecia) on leaf undersides',
            'Fruit may develop orange lesions',
            'Premature leaf drop',
            'Reduced fruit quality and yield'
        ],
        'treatment': [
            'Apply fungicides with myclobutanil or propiconazole',
            'Remove nearby juniper/cedar trees if possible',
            'Use preventive fungicide sprays in spring',
            'Collect and destroy fallen leaves'
        ],
        'seasons': ['Spring', 'Early Summer'],
        'peak_season': 'Spring',
        'prevention': [
            'Plant resistant apple varieties',
            'Remove alternate hosts (juniper/cedar) within 1-2 miles',
            'Apply preventive fungicide treatments'
        ],
        'severity': 'Medium',
        'spread_method': 'Spores from juniper/cedar hosts',
        'optimal_conditions': 'Cool, wet spring weather'
    },
    
    'Apple___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Regular pruning for air circulation',
            'Proper fertilization',
            'Adequate watering without overwatering',
            'Monitor for early disease signs'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, good air circulation'
    },
    
    'Apple___Apple_scab': {
        'symptoms': [
            'Olive-green to black spots on leaves',
            'Scaly, corky lesions on fruit',
            'Cracked and distorted fruit',
            'Premature leaf drop',
            'Reduced fruit quality and storage life'
        ],
        'treatment': [
            'Apply fungicides with captan, myclobutanil, or dodine',
            'Rake and destroy fallen leaves',
            'Prune for better air circulation',
            'Apply lime sulfur during dormant season'
        ],
        'seasons': ['Spring', 'Early Summer'],
        'peak_season': 'Spring',
        'prevention': [
            'Choose scab-resistant varieties',
            'Ensure good air circulation',
            'Avoid overhead irrigation',
            'Regular sanitation practices'
        ],
        'severity': 'High',
        'spread_method': 'Fungal spores via wind and splashing water',
        'optimal_conditions': 'Cool, wet spring weather (55-75°F)'
    },
    
    'Blueberry___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Maintain acidic soil (pH 4.5-5.5)',
            'Ensure good drainage',
            'Proper pruning for air circulation',
            'Regular monitoring for pests and diseases'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Acidic soil, consistent moisture, full sun to partial shade'
    },
    
    'Cherry_(including_sour)___Powdery_mildew': {
        'symptoms': [
            'White, powdery fungal growth on leaves',
            'Distorted or curled leaves',
            'Stunted shoot growth',
            'Reduced fruit production',
            'Leaves may turn yellow and drop'
        ],
        'treatment': [
            'Apply fungicides with myclobutanil, propiconazole, or sulfur',
            'Improve air circulation through pruning',
            'Remove infected plant parts',
            'Use horticultural oils'
        ],
        'seasons': ['Late Spring', 'Summer', 'Early Fall'],
        'peak_season': 'Summer',
        'prevention': [
            'Plant resistant varieties',
            'Ensure good air circulation',
            'Avoid overhead watering',
            'Regular pruning'
        ],
        'severity': 'Medium',
        'spread_method': 'Airborne fungal spores',
        'optimal_conditions': 'Warm days, cool nights, high humidity'
    },
    
    'Cherry_(including_sour)___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper pruning for air circulation',
            'Regular fertilization',
            'Adequate but not excessive watering',
            'Monitor for early disease signs'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, good air circulation'
    },
    
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'symptoms': [
            'Small, rectangular gray to tan lesions on leaves',
            'Lesions parallel to leaf veins',
            'Yellow halos around lesions',
            'Premature leaf death',
            'Reduced grain yield'
        ],
        'treatment': [
            'Apply fungicides with strobilurins or triazoles',
            'Crop rotation with non-host crops',
            'Use resistant hybrids',
            'Deep plowing to bury crop residue'
        ],
        'seasons': ['Mid to Late Summer'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Plant resistant varieties',
            'Crop rotation (2-3 years)',
            'Residue management',
            'Avoid late planting'
        ],
        'severity': 'Medium to High',
        'spread_method': 'Fungal spores via wind and rain splash',
        'optimal_conditions': 'Warm, humid conditions (80-85°F, high humidity)'
    },
    
    'Corn_(maize)___Common_rust_': {
        'symptoms': [
            'Small, round to oval rust-colored pustules on leaves',
            'Pustules on both sides of leaves',
            'Leaves may turn yellow and die prematurely',
            'Reduced photosynthesis',
            'Potential yield reduction'
        ],
        'treatment': [
            'Apply fungicides containing propiconazole or azoxystrobin',
            'Plant resistant hybrids',
            'Monitor weather conditions',
            'Early detection and treatment'
        ],
        'seasons': ['Mid Summer', 'Late Summer'],
        'peak_season': 'Mid to Late Summer',
        'prevention': [
            'Use resistant corn hybrids',
            'Monitor for early symptoms',
            'Proper field sanitation',
            'Avoid planting susceptible varieties in high-risk areas'
        ],
        'severity': 'Medium',
        'spread_method': 'Wind-borne fungal spores',
        'optimal_conditions': 'Cool, moist conditions (60-70°F)'
    },
    
    'Corn_(maize)___Northern_Leaf_Blight': {
        'symptoms': [
            'Large, elliptical gray-green lesions on lower leaves',
            'Lesions have dark borders',
            'Lesions may extend across entire leaf width',
            'Premature leaf death from bottom up',
            'Reduced grain fill and yield'
        ],
        'treatment': [
            'Apply fungicides with strobilurins or triazoles',
            'Use resistant hybrids',
            'Crop rotation',
            'Residue management'
        ],
        'seasons': ['Mid Summer', 'Late Summer'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Plant resistant varieties',
            'Crop rotation with non-host crops',
            'Till under crop residue',
            'Balanced fertilization'
        ],
        'severity': 'High',
        'spread_method': 'Wind and rain-dispersed spores',
        'optimal_conditions': 'Moderate temperatures (65-80°F) with high humidity'
    },
    
    'Corn_(maize)___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All growing seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper fertilization',
            'Adequate spacing between plants',
            'Regular monitoring for pests and diseases',
            'Good weed management'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, adequate moisture'
    },
    
    'Grape___Black_rot': {
        'symptoms': [
            'Circular brown spots on leaves with dark borders',
            'Black, mummified berries',
            'Brown lesions on shoots and petioles',
            'Premature fruit drop',
            'Reduced cluster quality'
        ],
        'treatment': [
            'Apply fungicides with captan, mancozeb, or tebuconazole',
            'Remove mummified berries and infected canes',
            'Improve air circulation through pruning',
            'Sanitation practices'
        ],
        'seasons': ['Spring', 'Summer'],
        'peak_season': 'Early to Mid Summer',
        'prevention': [
            'Plant resistant varieties',
            'Proper vine spacing',
            'Regular pruning for air circulation',
            'Remove infected plant material'
        ],
        'severity': 'High',
        'spread_method': 'Fungal spores via rain splash and wind',
        'optimal_conditions': 'Warm, wet weather (75-85°F)'
    },
    
    'Grape___Esca_(Black_Measles)': {
        'symptoms': [
            'Tiger stripe pattern on leaves (yellow and green bands)',
            'Dark spots on berries near harvest',
            'Stunted shoot growth',
            'Wilting and dieback of canes',
            'Internal wood decay'
        ],
        'treatment': [
            'Prune out infected wood',
            'Apply wound protectants after pruning',
            'Improve vine nutrition',
            'Trunk surgery in severe cases'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Proper pruning techniques',
            'Avoid large pruning wounds',
            'Maintain vine health',
            'Use clean pruning tools'
        ],
        'severity': 'High',
        'spread_method': 'Fungal spores enter through pruning wounds',
        'optimal_conditions': 'Stressed vines, large pruning wounds'
    },
    
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'symptoms': [
            'Dark brown to black angular spots on leaves',
            'Spots may have yellow halos',
            'Premature leaf drop',
            'Reduced photosynthesis',
            'Weakened vine vigor'
        ],
        'treatment': [
            'Apply copper-based fungicides',
            'Improve air circulation',
            'Remove infected leaves',
            'Reduce humidity around vines'
        ],
        'seasons': ['Late Summer', 'Fall'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Proper vine spacing',
            'Good canopy management',
            'Avoid overhead irrigation',
            'Regular monitoring'
        ],
        'severity': 'Medium',
        'spread_method': 'Rain splash and high humidity',
        'optimal_conditions': 'High humidity, warm temperatures'
    },
    
    'Grape___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper pruning and canopy management',
            'Good air circulation',
            'Balanced fertilization',
            'Regular monitoring for diseases'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, good air circulation'
    },
    
    'Orange___Haunglongbing_(Citrus_greening)': {
        'symptoms': [
            'Yellow mottle on leaves',
            'Asymmetrical blotchy yellowing',
            'Small, misshapen, bitter fruit',
            'Twig dieback',
            'Tree decline and death'
        ],
        'treatment': [
            'Remove infected trees immediately',
            'Control Asian citrus psyllid vector',
            'Apply insecticides for psyllid control',
            'No cure available - focus on prevention'
        ],
        'seasons': ['All seasons'],
        'peak_season': 'Year-round',
        'prevention': [
            'Control psyllid vector',
            'Quarantine measures',
            'Use certified disease-free nursery stock',
            'Regular inspection and early detection'
        ],
        'severity': 'Very High',
        'spread_method': 'Asian citrus psyllid vector',
        'optimal_conditions': 'Presence of psyllid vector'
    },
    
    'Peach___Bacterial_spot': {
        'symptoms': [
            'Small, dark spots on leaves with yellow halos',
            'Spots on fruit causing cracking and scarring',
            'Twig cankers',
            'Premature fruit drop',
            'Reduced fruit quality'
        ],
        'treatment': [
            'Apply copper-based bactericides',
            'Use streptomycin during bloom',
            'Prune for air circulation',
            'Remove infected plant material'
        ],
        'seasons': ['Spring', 'Summer'],
        'peak_season': 'Late Spring to Early Summer',
        'prevention': [
            'Plant resistant varieties',
            'Avoid overhead irrigation',
            'Proper pruning for air circulation',
            'Copper sprays during dormant season'
        ],
        'severity': 'Medium to High',
        'spread_method': 'Rain splash and wind',
        'optimal_conditions': 'Warm, wet weather'
    },
    
    'Peach___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper pruning for air circulation',
            'Regular fertilization',
            'Adequate watering',
            'Monitor for early disease signs'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, good air circulation'
    },
    
    'Pepper,_bell___Bacterial_spot': {
        'symptoms': [
            'Small, dark spots on leaves with yellow halos',
            'Raised, scab-like spots on fruit',
            'Leaf yellowing and drop',
            'Stunted plant growth',
            'Reduced fruit quality and yield'
        ],
        'treatment': [
            'Apply copper-based bactericides',
            'Use streptomycin if available',
            'Remove infected plants',
            'Improve air circulation'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Mid to Late Summer',
        'prevention': [
            'Use certified disease-free seeds',
            'Avoid overhead watering',
            'Crop rotation',
            'Plant resistant varieties'
        ],
        'severity': 'Medium to High',
        'spread_method': 'Rain splash, contaminated tools, insects',
        'optimal_conditions': 'Warm, humid conditions'
    },
    
    'Pepper,_bell___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All growing seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper spacing for air circulation',
            'Regular fertilization',
            'Consistent watering',
            'Monitor for early disease signs'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, consistent moisture'
    },
    
    'Potato___Early_blight': {
        'symptoms': [
            'Dark spots with concentric rings on lower leaves',
            'Yellow halos around spots',
            'Dark lesions on tubers',
            'Premature leaf yellowing and drop',
            'Reduced tuber quality'
        ],
        'treatment': [
            'Apply fungicides with chlorothalonil or mancozeb',
            'Remove infected plant debris',
            'Improve air circulation',
            'Avoid overhead watering'
        ],
        'seasons': ['Mid Summer', 'Late Summer'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Plant certified seed potatoes',
            'Crop rotation',
            'Proper spacing',
            'Regular fungicide applications'
        ],
        'severity': 'Medium to High',
        'spread_method': 'Wind-borne spores and rain splash',
        'optimal_conditions': 'Warm, humid weather (75-85°F)'
    },
    
    'Potato___Late_blight': {
        'symptoms': [
            'Water-soaked lesions on leaves',
            'White fuzzy growth on leaf undersides',
            'Brown to black lesions on tubers',
            'Rapid plant collapse',
            'Foul odor from infected tubers'
        ],
        'treatment': [
            'Apply fungicides with metalaxyl or mancozeb',
            'Destroy infected plants immediately',
            'Improve drainage',
            'Avoid overhead irrigation'
        ],
        'seasons': ['Late Summer', 'Fall'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Use certified seed potatoes',
            'Plant resistant varieties',
            'Ensure good drainage',
            'Monitor weather conditions'
        ],
        'severity': 'Very High',
        'spread_method': 'Wind-borne spores, contaminated soil',
        'optimal_conditions': 'Cool, wet weather (60-70°F)'
    },
    
    'Potato___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All growing seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Use certified seed potatoes',
            'Proper crop rotation',
            'Good drainage',
            'Regular monitoring for diseases'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, cool temperatures, consistent moisture'
    },
    
    'Raspberry___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper pruning for air circulation',
            'Good drainage',
            'Regular fertilization',
            'Monitor for early disease signs'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, partial shade to full sun'
    },
    
    'Soybean___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All growing seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Crop rotation',
            'Proper plant spacing',
            'Good weed management',
            'Regular monitoring for pests and diseases'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, adequate moisture'
    },
    
    'Squash___Powdery_mildew': {
        'symptoms': [
            'White, powdery fungal growth on leaves',
            'Yellow spots on upper leaf surface',
            'Stunted plant growth',
            'Reduced fruit production',
            'Premature plant death'
        ],
        'treatment': [
            'Apply fungicides with sulfur or potassium bicarbonate',
            'Improve air circulation',
            'Remove infected leaves',
            'Use horticultural oils'
        ],
        'seasons': ['Mid Summer', 'Late Summer', 'Fall'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Plant resistant varieties',
            'Ensure good air circulation',
            'Avoid overhead watering',
            'Regular monitoring'
        ],
        'severity': 'Medium',
        'spread_method': 'Airborne fungal spores',
        'optimal_conditions': 'Warm days, cool nights, high humidity'
    },
    
    'Strawberry___Leaf_scorch': {
        'symptoms': [
            'Purple to reddish-brown spots on leaves',
            'Spots have white to gray centers',
            'Premature leaf browning and death',
            'Reduced plant vigor',
            'Decreased fruit production'
        ],
        'treatment': [
            'Apply fungicides with captan or myclobutanil',
            'Remove infected leaves and debris',
            'Improve air circulation',
            'Avoid overhead watering'
        ],
        'seasons': ['Spring', 'Summer'],
        'peak_season': 'Late Spring to Early Summer',
        'prevention': [
            'Plant resistant varieties',
            'Proper plant spacing',
            'Good sanitation practices',
            'Drip irrigation instead of overhead'
        ],
        'severity': 'Medium',
        'spread_method': 'Fungal spores via rain splash',
        'optimal_conditions': 'Warm, humid conditions'
    },
    
    'Strawberry___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper plant spacing',
            'Good drainage',
            'Regular fertilization',
            'Monitor for early disease signs'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun to partial shade'
    },
    
    'Tomato___Bacterial_spot': {
        'symptoms': [
            'Small, dark spots on leaves with yellow halos',
            'Raised, scab-like spots on fruit',
            'Leaf yellowing and drop',
            'Cracking and scarring of fruit',
            'Reduced plant vigor'
        ],
        'treatment': [
            'Apply copper-based bactericides',
            'Use streptomycin during early stages',
            'Remove infected plants',
            'Improve air circulation'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Mid to Late Summer',
        'prevention': [
            'Use certified disease-free seeds',
            'Avoid overhead watering',
            'Crop rotation',
            'Plant resistant varieties'
        ],
        'severity': 'Medium to High',
        'spread_method': 'Rain splash, contaminated tools, insects',
        'optimal_conditions': 'Warm, humid conditions'
    },
    
    'Tomato___Early_blight': {
        'symptoms': [
            'Dark spots with concentric rings on lower leaves',
            'Yellow halos around spots',
            'Stem lesions near soil line',
            'Fruit spots with dark, sunken centers',
            'Premature leaf drop'
        ],
        'treatment': [
            'Apply fungicides with chlorothalonil or mancozeb',
            'Remove infected plant debris',
            'Mulch around plants',
            'Improve air circulation'
        ],
        'seasons': ['Mid Summer', 'Late Summer'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Crop rotation',
            'Proper plant spacing',
            'Avoid overhead watering',
            'Regular fungicide applications'
        ],
        'severity': 'Medium to High',
        'spread_method': 'Wind-borne spores, rain splash',
        'optimal_conditions': 'Warm, humid weather (75-85°F)'
    },
    
    'Tomato___Late_blight': {
        'symptoms': [
            'Water-soaked lesions on leaves',
            'White fuzzy growth on leaf undersides',
            'Brown lesions on stems and fruit',
            'Rapid plant collapse',
            'Foul odor from infected tissues'
        ],
        'treatment': [
            'Apply fungicides with metalaxyl or copper',
            'Remove infected plants immediately',
            'Improve air circulation',
            'Avoid overhead irrigation'
        ],
        'seasons': ['Late Summer', 'Fall'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Plant resistant varieties',
            'Ensure good drainage',
            'Monitor weather conditions',
            'Use certified disease-free transplants'
        ],
        'severity': 'Very High',
        'spread_method': 'Wind-borne spores, contaminated soil',
        'optimal_conditions': 'Cool, wet weather (60-70°F)'
    },
    
    'Tomato___Leaf_Mold': {
        'symptoms': [
            'Yellow spots on upper leaf surface',
            'Fuzzy olive-green to brown growth on leaf undersides',
            'Leaf curling and yellowing',
            'Premature leaf drop',
            'Reduced fruit production'
        ],
        'treatment': [
            'Improve ventilation in greenhouses',
            'Apply fungicides with chlorothalonil',
            'Remove infected leaves',
            'Reduce humidity levels'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Mid to Late Summer',
        'prevention': [
            'Plant resistant varieties',
            'Ensure good air circulation',
            'Control humidity levels',
            'Avoid overhead watering'
        ],
        'severity': 'Medium',
        'spread_method': 'Airborne spores, high humidity',
        'optimal_conditions': 'High humidity (>85%), moderate temperatures'
    },
    
    'Tomato___Septoria_leaf_spot': {
        'symptoms': [
            'Small, circular spots with dark borders on leaves',
            'Spots have gray or tan centers with tiny black dots',
            'Lower leaves affected first',
            'Progressive yellowing and defoliation',
            'Reduced fruit quality'
        ],
        'treatment': [
            'Apply fungicides with chlorothalonil or copper',
            'Remove infected lower leaves',
            'Mulch around plants',
            'Improve air circulation'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Mid to Late Summer',
        'prevention': [
            'Crop rotation',
            'Avoid overhead watering',
            'Proper plant spacing',
            'Use certified disease-free seeds'
        ],
        'severity': 'Medium',
        'spread_method': 'Rain splash, contaminated debris',
        'optimal_conditions': 'Warm, wet weather (75-85°F)'
    },
    
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'symptoms': [
            'Fine webbing on leaves and stems',
            'Yellow stippling on leaves',
            'Bronzing of leaves',
            'Premature leaf drop',
            'Stunted plant growth'
        ],
        'treatment': [
            'Apply miticides or insecticidal soaps',
            'Increase humidity around plants',
            'Remove severely infested leaves',
            'Use predatory mites as biological control'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Hot, dry summer periods',
        'prevention': [
            'Maintain adequate moisture',
            'Regular monitoring for early detection',
            'Avoid over-fertilization with nitrogen',
            'Encourage beneficial insects'
        ],
        'severity': 'Medium',
        'spread_method': 'Wind dispersal, contaminated tools',
        'optimal_conditions': 'Hot, dry conditions'
    },
    
    'Tomato___Target_Spot': {
        'symptoms': [
            'Small, dark spots with concentric rings on leaves',
            'Brown lesions on stems and fruit',
            'Yellow halos around leaf spots',
            'Premature defoliation',
            'Reduced fruit quality'
        ],
        'treatment': [
            'Apply fungicides with azoxystrobin or chlorothalonil',
            'Remove infected plant debris',
            'Improve air circulation',
            'Rotate crops'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Late Summer',
        'prevention': [
            'Crop rotation',
            'Avoid overhead irrigation',
            'Plant resistant varieties',
            'Proper sanitation'
        ],
        'severity': 'Medium',
        'spread_method': 'Rain splash, wind-borne spores',
        'optimal_conditions': 'Warm, humid conditions'
    },
    
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'symptoms': [
            'Upward curling and yellowing of leaves',
            'Stunted plant growth',
            'Reduced fruit set',
            'Small, poor-quality fruit',
            'Interveinal yellowing'
        ],
        'treatment': [
            'Remove infected plants immediately',
            'Control whitefly vectors',
            'Use reflective mulches',
            'Apply insecticides for whitefly control'
        ],
        'seasons': ['Summer', 'Fall'],
        'peak_season': 'Hot summer months',
        'prevention': [
            'Use virus-resistant varieties',
            'Control whitefly populations',
            'Use physical barriers (row covers)',
            'Remove infected plants promptly'
        ],
        'severity': 'High',
        'spread_method': 'Whitefly vector transmission',
        'optimal_conditions': 'Hot weather, presence of whiteflies'
    },
    
    'Tomato___Tomato_mosaic_virus': {
        'symptoms': [
            'Mottled light and dark green patterns on leaves',
            'Stunted plant growth',
            'Distorted or malformed leaves',
            'Reduced fruit production',
            'Fruit may show color variations',
            'Plant yellowing and decline'
        ],
        'treatment': [
            'Remove infected plants immediately',
            'Disinfect tools and hands',
            'No chemical cure available',
            'Focus on prevention and control'
        ],
        'seasons': ['All growing seasons'],
        'peak_season': 'Throughout growing season',
        'prevention': [
            'Use certified virus-free seeds and transplants',
            'Avoid smoking around plants',
            'Disinfect tools between plants',
            'Control aphid vectors',
            'Remove infected plants promptly'
        ],
        'severity': 'High',
        'spread_method': 'Mechanical transmission, contaminated tools, aphids',
        'optimal_conditions': 'Any conditions - spreads through contact'
    },
    
    'Tomato___healthy': {
        'symptoms': ['No disease symptoms present'],
        'treatment': ['Maintain regular care practices'],
        'seasons': ['All growing seasons'],
        'peak_season': 'Growing season',
        'prevention': [
            'Proper plant spacing for air circulation',
            'Regular fertilization',
            'Consistent watering (avoid overhead)',
            'Monitor for early disease signs',
            'Crop rotation',
            'Use certified disease-free seeds'
        ],
        'severity': 'None',
        'spread_method': 'N/A',
        'optimal_conditions': 'Well-drained soil, full sun, consistent moisture, good air circulation'
    }
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        top_class = torch.argmax(prob).item()
        return class_names[top_class], round(prob[0][top_class].item() * 100, 2)

def get_disease_info(disease):
    return plant_disease_database.get(disease, {
        'symptoms': ['Not available'],
        'treatment': ['Not available'],
        'season': ['Unknown']
    })







#1. `import torch` → Import PyTorch for deep learning operations.
#2. `import torch.nn.functional as F` → Import PyTorch functional API for operations like softmax.
#3. `from torchvision import transforms` → Import tools to preprocess images for the model.
#4. `from PIL import Image` → Import Python Imaging Library to open and manipulate images.
#5. `from timm import create_model` → Import function to create pre-defined deep learning models like DenseNet.
#6. `# Load model` → Comment describing the next lines.
#7. `model = create_model('densenet121', pretrained=False, num_classes=38)` → Create DenseNet-121 model with 38 output classes, no pretrained weights.
#8. `model.load_state_dict(torch.load('model/densenet_weights.pth', map_location='cpu'))` → Load saved model weights to CPU.
#9. `model.eval()` → Set model to evaluation mode (disable dropout/batch norm updates).
#11. `class_names = [` → Start of dictionary containing disease information.
#12. `'Tomato___healthy': {` → Define entry for healthy tomato class.
#13. `'symptoms': ['No disease symptoms present'],` → List of symptoms.
#14. `'treatment': ['Maintain regular care practices'],` → Treatment methods.
#15. `'seasons': ['All growing seasons'],` → Seasons for this class.
#16. `'peak_season': 'Growing season',` → Peak growing season.
#17. `'prevention': [` → Prevention methods list.
#18. `'Proper plant spacing for air circulation',`
#19. `'Regular fertilization',`
#20. `'Consistent watering (avoid overhead)',`
#21. `'Monitor for early disease signs',`
#22. `'Crop rotation',`
#23. `'Use certified disease-free seeds'`
#24. `],`
#25. `'severity': 'None',` → Severity of disease.
#26. `'spread_method': 'N/A',` → How disease spreads.
#27. `'optimal_conditions': 'Well-drained soil, full sun, consistent moisture, good air circulation'` → Best growth conditions.
#28. `}` → Close dictionary for class.
#29. `]` → Close class\_names list.
#30. `# Preprocessing` → Comment describing next section.
#31. `transform = transforms.Compose([` → Define preprocessing pipeline.
#32. `transforms.Resize((224, 224)),` → Resize image to 224x224 (DenseNet input).
#33. `transforms.ToTensor(),` → Convert image to PyTorch tensor.
#34. `transforms.Normalize([0.485, 0.456, 0.406],` → Normalize image using ImageNet mean.
#35. `[0.229, 0.224, 0.225])` → Normalize using ImageNet std.
#36. `])` → Close transform pipeline.
#37. `def predict_image(img_path):` → Function to predict disease from image path.
#38. `img = Image.open(img_path).convert('RGB')` → Open image and ensure RGB format.
#39. `input_tensor = transform(img).unsqueeze(0)` → Preprocess image and add batch dimension.
#40. `with torch.no_grad():` → Disable gradient computation (inference mode).
#41. `output = model(input_tensor)` → Forward pass through model.
#42. `prob = F.softmax(output, dim=1)` → Convert logits to probabilities.
#43. `top_class = torch.argmax(prob).item()` → Get class index with highest probability.
#44. `return class_names[top_class], round(prob[0][top_class].item() * 100, 2)` → Return class info and confidence %.
#45. `def get_disease_info(disease):` → Function to get additional disease info.
#46. `return plant_disease_database.get(disease, {` → Fetch from database or return default.
#47. `'symptoms': ['Not available'],` → Default symptom.
#48. `'treatment': ['Not available'],` → Default treatment.
#49. `'season': ['Unknown']` → Default season.
#50. `})` → Close default dictionary.


