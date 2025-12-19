import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

class ImageClassifier:
    def __init__(self):
        print("Loading ResNet18 AI model for ANIMAL detection...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        
        # Load ImageNet labels
        self.all_labels = self._get_labels()
        
        # ANIMAL-SPECIFIC CATEGORIES - Focus only on animals
        self.animal_categories = {
            'dog': self._get_animal_labels('dog'),
            'cat': self._get_animal_labels('cat'),
            'bird': self._get_animal_labels('bird'),
            'other_animal': self._get_other_animal_labels(),
            'non_animal': []  # For things that aren't animals
        }
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        print("Animal Classifier ready!")
        print(f"Target categories: {list(self.animal_categories.keys())}")

    def _get_labels(self):
        """Download ImageNet labels"""
        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            with urllib.request.urlopen(LABELS_URL) as url:
                return json.loads(url.read().decode())
        except:
            # Fallback to local labels if no internet
            return [f"class_{i}" for i in range(1000)]

    def _get_animal_labels(self, animal_type):
        """Get ImageNet labels for specific animal types"""
        animal_keywords = {
            'dog': [
                'greyhound', 'golden retriever', 'Labrador retriever', 'German shepherd',
                'pug', 'chihuahua', 'beagle', 'bulldog', 'husky', 'dalmatian',
                'Great Dane', 'boxer', 'Doberman', 'rottweiler', 'schnauzer',
                'Shetland sheepdog', 'whippet', 'bloodhound', 'malamute', 'papillon',
                'Shih-Tzu', 'Afghan hound', 'basset', 'cocker spaniel', 'collie',
                'poodle', 'dingo', 'dhole', 'African hunting dog'
            ],
            'cat': [
                'tabby', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'tiger cat',
                'cat'
            ],
            'bird': [
                'robin', 'junco', 'brambling', 'sparrow', 'eagle', 'vulture',
                'peacock', 'flamingo', 'ostrich', 'penguin', 'bald eagle',
                'golden eagle', 'black grouse', 'ptarmigan', 'ruffed grouse',
                'prairie chicken', 'quail', 'partridge', 'African grey',
                'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater',
                'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser',
                'goose', 'black swan', 'white stork', 'black stork', 'spoonbill',
                'little blue heron', 'American egret', 'bittern', 'crane', 'limpkin',
                'rail', 'coot', 'bustard', 'ruddy turnstone', 'red-backed sandpiper',
                'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin',
                'albatross'
            ]
        }
        return animal_keywords.get(animal_type, [])

    def _get_other_animal_labels(self):
        """Get labels for other animals (not dog/cat/bird)"""
        return [
            # Big cats
            'tiger', 'bengal tiger', 'siberian tiger', 'amur tiger', 'cheetah', 'lion', 'african lion', 'asian lion',
            'snow leopard', 'irbis', 'jaguar', 'cougar', 'mountain lion', 'puma', 'panther',
            'lynx', 'leopard', 'african leopard', 'clouded leopard', 'amur leopard',
            'ocelot', 'caracal', 'serval', 'linsang', 'binturong',
            
            # Bears
            'american black bear', 'black bear', 'brown bear', 'grizzly bear', 'grizzly',
            'polar bear', 'ice bear', 'ursus maritimus', 'sloth bear', 'honey bear',
            'sun bear', 'panda bear', 'giant panda',
            
            # Canines (besides dogs)
            'wolf', 'gray wolf', 'grey wolf', 'arctic wolf', 'red wolf',
            'fox', 'red fox', 'arctic fox', 'fennec', 'kit fox', 'grey fox',
            'jackal', 'coyote',
            
            # Primates
            'orangutan', 'orang-utan', 'orang utan', 'gorilla', 'lowland gorilla', 'mountain gorilla',
            'chimpanzee', 'chimp', 'pan troglodytes', 'gibbon', 'siamang', 'lar gibbon',
            'baboon', 'hamadryas', 'mandrill', 'drill', 'macaque', 'rhesus', 'rhesus monkey',
            'vervet', 'vervet monkey', 'langur', 'colobus', 'colobus monkey', 'guenon',
            'patas', 'capuchin', 'howler', 'howler monkey', 'spider monkey', 'squirrel monkey',
            'titi', 'marmoset', 'tamarin', 'indri', 'lemur', 'ring-tailed lemur',
            'aye-aye', 'aye aye', 'tarsier', 'tarsius', 'loris', 'slow loris', 'slender loris',
            'galago', 'bushbaby', 'bush baby',
            
            # Hyenas
            'hyena', 'hyaena', 'spotted hyena', 'striped hyena',
            
            # Elephants
            'Indian elephant', 'African elephant', 'Asian elephant', 'elephant',
            
            # Marine mammals
            'penguin', 'emperor penguin', 'rockhopper', 'adelie', 'king penguin',
            'whale', 'blue whale', 'grey whale', 'gray whale', 'killer whale', 'orca', 'humpback',
            'dolphin', 'bottlenose dolphin', 'bottlenose', 'spinner dolphin',
            'dugong', 'sea lion', 'seal', 'fur seal', 'leopard seal',
            'walrus', 'sea otter', 'otter', 'river otter',
            'manatee', 'sea cow', 'narwhal', 'beluga',
            
            # Snakes
            'python', 'ball python', 'reticulated python', 'burmese python',
            'cobra', 'king cobra', 'indian cobra', 'black cobra', 'spitting cobra',
            'mamba', 'green mamba', 'black mamba',
            'boa', 'boa constrictor', 'anaconda', 'green anaconda', 'yellow anaconda',
            'rattlesnake', 'diamondback', 'diamondback rattlesnake', 'sidewinder',
            'viper', 'pit viper', 'horned viper',
            'sea snake', 'sea krait', 'krait',
            'corn snake', 'milk snake', 'king snake', 'water snake', 'garter snake',
            'hognose snake', 'black snake', 'green snake',
            'thunder snake', 'ringneck snake',
            'vine snake', 'night snake', 'rock python',
            'spitting snake',
            
            # Lizards
            'komodo dragon', 'komodo', 'monitor', 'monitor lizard', 'nile monitor',
            'iguana', 'green iguana', 'common iguana',
            'chameleon', 'veiled chameleon', 'panther chameleon', 'jackson chameleon', 'crested chameleon',
            'african chameleon', 'gecko', 'leopard gecko', 'day gecko', 'crested gecko',
            'frilled lizard', 'frilled', 'agama', 'bearded agama',
            'basilisk', 'horned lizard', 'gila monster', 'beaded lizard',
            'skink', 'slow worm', 'legless lizard', 'alligator lizard',
            'green lizard', 'wall lizard', 'anole', 'whiptail',
            
            # Turtles and Crocodilians
            'tortoise', 'giant tortoise', 'galapagos tortoise', 'aldabra tortoise',
            'sea turtle', 'green sea turtle', 'loggerhead', 'loggerhead turtle',
            'hawksbill turtle', 'hawksbill', 'leatherback turtle', 'leatherback', 'olive ridley',
            'turtle', 'box turtle', 'painted turtle', 'red-eared slider', 'slider',
            'snapping turtle', 'terrapin', 'mud turtle',
            'crocodile', 'nile crocodile', 'saltwater crocodile', 'saltie', 'american crocodile',
            'alligator', 'american alligator', 'gator',
            'caiman', 'spectacled caiman', 'black caiman',
            'gharial', 'false gharial',
            
            # Amphibians
            'poison dart frog', 'poison frog', 'dart frog',
            'tree frog', 'red-eyed tree frog', 'red-eyed',
            'bullfrog', 'american bullfrog', 'bull frog',
            'frog', 'leopard frog', 'green frog', 'chorus frog', 'spring peeper',
            'toad', 'cane toad', 'bufo', 'american toad', 'true toad',
            'axolotl', 'salamander', 'tiger salamander', 'spotted salamander',
            'fire salamander', 'european fire salamander',
            'hellbender', 'giant salamander', 'chinese salamander', 'japanese salamander',
            'newt', 'alpine newt', 'smooth newt', 'common newt', 'eft',
            'mud puppy', 'siren', 'caecilian',
            
            # Ungulates (hoofed animals)
            'zebra', 'plains zebra', 'mountain zebra', "grevy's zebra",
            'giraffe', 'okapi',
            'hippopotamus', 'hippo',
            'rhinoceros', 'rhino', 'white rhino', 'black rhino', 'sumatran rhino',
            'deer', 'red deer', 'mule deer', 'sika deer',
            'elk', 'moose', 'reindeer', 'caribou',
            'antelope', 'pronghorn', 'gazelle', 'springbok', 'impala', 'kudu',
            'eland', 'wildebeest', 'gnu', 'nyala', 'oryx', 'addax',
            'gemsbuck', 'waterbuck', 'bushbuck', 'duiker',
            'buffalo', 'water buffalo', 'bison', 'american bison', 'european bison',
            'camel', 'dromedary', 'arabian camel', 'bactrian camel',
            'llama', 'alpaca', 'guanaco', 'vicuna',
            'sheep', 'lamb', 'ram', 'bighorn', 'bighorn sheep',
            'goat', 'kid', 'mountain goat', 'ibex',
            'horse', 'pony', 'mare', 'stallion', 'colt', 'foal',
            'donkey', 'mule', 'ass',
            'cattle', 'cow', 'bull', 'ox', 'holstein', 'hereford',
            'pig', 'hog', 'wild boar', 'warthog', 'babirusa', 'peccary', 'javelina',
            'ox', 'yak', 'musk ox', 'chevrotain', 'takin',
            'sorrel',
            
            # Other small mammals
            'red panda', 'raccoon', 'skunk', 'weasel', 'mink',
            'polecat', 'black-footed ferret', 'ferret', 'badger', 'armadillo',
            'three-toed sloth', 'two-toed sloth', 'sloth', 'porcupine', 'quill pig',
            'beaver', 'hamster', 'guinea pig', 'hedgehog', 'mole', 'shrew',
            'flying squirrel', 'fox squirrel', 'squirrel', 'marmot',
            'prairie dog', 'ground squirrel', 'chipmunk',
            'chinchilla', 'paca', 'agouti', 'nutria', 'capybara', 'desman',
            
            # Marsupials
            'koala', 'wombat', 'wallaby', 'kangaroo', 'red kangaroo', 'grey kangaroo',
            'tree kangaroo', 'echidna', 'platypus', 'tasmanian devil',
            'sugar glider', 'feathertail glider', 'opossum', 'possum',
            'quokka', 'numbat', 'burrowing bettong',
            
            # Misc exotic mammals
            'aardvark', 'anteater', 'ant eater', 'pangolin', 'hyrax',
            'rock hyrax', 'tree hyrax', 'yellow-spotted hyrax',
            'colugo', 'flying lemur', 'flying dragon', 'scaly-tailed flying dragon',
            'emu', 'cassowary', 'kiwi', 'rhea',
            'tree shrew', 'tupaia', 'tupayo', 'treeshrew'
        ]

    def _generalize_animal_label(self, specific_label):
        """
        Convert specific breed/species names to general animal categories
        Example: 'golden retriever' -> 'Dog', 'Persian cat' -> 'Cat'
        """
        label_lower = specific_label.lower()
        
        # Dog breeds -> Dog
        dog_keywords = ['retriever', 'shepherd', 'pug', 'chihuahua', 'beagle', 'bulldog', 
                       'husky', 'dalmatian', 'dane', 'boxer', 'doberman', 'rottweiler',
                       'schnauzer', 'sheepdog', 'whippet', 'bloodhound', 'malamute',
                       'papillon', 'shih', 'hound', 'basset', 'spaniel', 'collie',
                       'poodle', 'terrier', 'dingo', 'dhole']
        if any(keyword in label_lower for keyword in dog_keywords):
            return "Dog"
        
        # Domestic cats -> Cat (but not big cats)
        if any(keyword in label_lower for keyword in ['tabby', 'persian cat', 'siamese', 'egyptian cat', 'tiger cat']):
            return "Cat"
        if label_lower == 'cat':
            return "Cat"
        
        # Big cats -> specific names
        big_cats = {
            'tiger': 'Tiger', 'lion': 'Lion', 'cheetah': 'Cheetah',
            'leopard': 'Leopard', 'jaguar': 'Jaguar', 'cougar': 'Cougar',
            'lynx': 'Lynx', 'snow leopard': 'Snow Leopard'
        }
        for keyword, general_name in big_cats.items():
            if keyword in label_lower:
                return general_name
        
        # Birds: keep only key species as specific labels; everything else collapses to Bird
        if any(keyword in label_lower for keyword in ['duck', 'mallard', 'wood duck', 'teal']):
            return "Duck"
        if any(keyword in label_lower for keyword in ['swan', 'black swan', 'mute swan']):
            return "Swan"
        if any(keyword in label_lower for keyword in ['goose', 'canada goose', 'snow goose']):
            return "Goose"
        if any(keyword in label_lower for keyword in ['peacock', 'peafowl', 'indian peafowl']):
            return "Peacock"
        if any(keyword in label_lower for keyword in ['flamingo', 'pink flamingo']):
            return "Flamingo"
        if 'ostrich' in label_lower:
            return "Ostrich"
        if any(keyword in label_lower for keyword in ['chicken', 'rooster', 'hen', 'chicken rooster']):
            return "Chicken"
        if 'turkey' in label_lower:
            return "Turkey"

        bird_keywords = [
            'bird', 'eagle', 'hawk', 'owl', 'penguin', 'parrot', 'macaw', 'cockatoo', 'lorikeet',
            'parakeet', 'hornbill', 'hummingbird', 'toucan', 'stork', 'egret', 'crane', 'pelican',
            'duck', 'swan', 'goose', 'peacock', 'flamingo', 'ostrich', 'chicken', 'turkey', 'emu',
            'cassowary', 'kiwi', 'rhea', 'crow', 'raven', 'jay', 'magpie', 'falcon', 'vulture', 'robin',
            'woodpecker', 'albatross', 'puffin', 'gull', 'tern', 'heron', 'sandpiper', 'avocet', 'plover',
            'loon', 'cormorant'
        ]
        if any(keyword in label_lower for keyword in bird_keywords):
            return "Bird"
        
        # Elephants
        if 'elephant' in label_lower:
            return "Elephant"
        
        # Pandas (special case)
        if 'panda' in label_lower:
            return "Panda"
        
        # Bears (all non-panda bears collapse to Bear)
        if 'bear' in label_lower and 'teddy' not in label_lower:
            return "Bear"
        
        # Primates
        if any(keyword in label_lower for keyword in ['monkey', 'ape', 'gorilla', 'chimpanzee', 
                                                       'orangutan', 'gibbon', 'baboon', 'macaque',
                                                       'lemur', 'marmoset', 'capuchin']):
            return "Monkey/Primate"
        
        # Marine mammals
        if any(keyword in label_lower for keyword in ['whale', 'dolphin', 'seal', 'walrus', 
                                                       'sea lion', 'dugong']):
            return "Marine Mammal"
        
        # Reptiles
        if any(keyword in label_lower for keyword in ['snake', 'lizard', 'iguana', 'chameleon',
                                                       'gecko', 'crocodile', 'alligator', 'turtle',
                                                       'tortoise', 'terrapin']):
            return "Reptile"
        
        # Amphibians
        if any(keyword in label_lower for keyword in ['frog', 'toad', 'salamander', 'newt']):
            return "Amphibian"
        
        # Farm animals - more specific labels for common farm animals
        if any(keyword in label_lower for keyword in ['cow', 'bull', 'cattle', 'holstein', 'hereford']):
            return "Cow"
        if any(keyword in label_lower for keyword in ['sheep', 'lamb']):
            return "Sheep"
        if any(keyword in label_lower for keyword in ['goat', 'kid']):
            return "Goat"
        if any(keyword in label_lower for keyword in ['horse', 'pony', 'mare', 'stallion', 'colt', 'foal']):
            return "Horse"
        if any(keyword in label_lower for keyword in ['pig', 'hog', 'swine']):
            return "Pig"
        if any(keyword in label_lower for keyword in ['donkey', 'mule', 'ass']):
            return "Donkey"
        
        # Wild hoofed animals
        if any(keyword in label_lower for keyword in ['deer', 'elk', 'moose', 'reindeer', 'caribou']):
            return "Deer"
        if 'zebra' in label_lower:
            return "Zebra"
        if 'giraffe' in label_lower:
            return "Giraffe"
        if any(keyword in label_lower for keyword in ['gazelle', 'antelope', 'impala']):
            return "Antelope"
        if 'buffalo' in label_lower or 'bison' in label_lower:
            return "Buffalo"
        if any(keyword in label_lower for keyword in ['camel', 'dromedary']):
            return "Camel"
        if 'llama' in label_lower or 'alpaca' in label_lower:
            return "Llama"
        if 'rhino' in label_lower:
            return "Rhinoceros"
        if 'hippopotamus' in label_lower or 'hippo' in label_lower:
            return "Hippopotamus"
        if 'warthog' in label_lower:
            return "Warthog"
        
        # Wolves and canines (besides dogs)
        if any(keyword in label_lower for keyword in ['wolf', 'gray wolf', 'grey wolf', 'arctic wolf', 'red wolf']):
            return "Wolf"
        if 'coyote' in label_lower:
            return "Coyote"
        if 'jackal' in label_lower:
            return "Jackal"
        if any(keyword in label_lower for keyword in ['fox', 'red fox', 'arctic fox', 'fennec', 'kit fox', 'grey fox']):
            return "Fox"
        
        # Big cats
        if any(keyword in label_lower for keyword in ['tiger', 'bengal', 'siberian', 'amur tiger']):
            return "Tiger"
        if any(keyword in label_lower for keyword in ['lion', 'african lion', 'asian lion']):
            return "Lion"
        if any(keyword in label_lower for keyword in ['leopard', 'clouded leopard', 'amur leopard', 'african leopard']):
            return "Leopard"
        if 'cheetah' in label_lower:
            return "Cheetah"
        if any(keyword in label_lower for keyword in ['snow leopard', 'irbis']):
            return "Snow Leopard"
        if 'jaguar' in label_lower:
            return "Jaguar"
        if any(keyword in label_lower for keyword in ['cougar', 'mountain lion', 'puma', 'panther']):
            return "Cougar"
        if 'lynx' in label_lower:
            return "Lynx"
        if 'ocelot' in label_lower:
            return "Ocelot"
        if 'caracal' in label_lower:
            return "Caracal"
        if 'serval' in label_lower:
            return "Serval"
        if 'binturong' in label_lower:
            return "Binturong"
        
        # Primates
        if any(keyword in label_lower for keyword in ['orangutan', 'orang-utan', 'orang utan']):
            return "Orangutan"
        if any(keyword in label_lower for keyword in ['gorilla', 'lowland', 'mountain gorilla', 'western']):
            return "Gorilla"
        if any(keyword in label_lower for keyword in ['chimpanzee', 'chimp', 'pan troglodytes']):
            return "Chimpanzee"
        if any(keyword in label_lower for keyword in ['gibbon', 'siamang', 'lar gibbon', 'white-handed']):
            return "Gibbon"
        if any(keyword in label_lower for keyword in ['baboon', 'hamadryas', 'mandrill', 'drill']):
            return "Baboon"
        if any(keyword in label_lower for keyword in ['monkey', 'macaque', 'rhesus', 'vervet', 'langur',
                                                       'colobus', 'guenon', 'mangabey', 'capuchin',
                                                       'howler', 'spider', 'squirrel', 'marmoset', 'tamarin']):
            return "Monkey"
        if any(keyword in label_lower for keyword in ['lemur', 'ring-tailed', 'lemure', 'ringtail']):
            return "Lemur"
        if any(keyword in label_lower for keyword in ['aye-aye', 'aye aye']):
            return "Aye-Aye"
        if 'tarsier' in label_lower or 'tarsius' in label_lower:
            return "Tarsier"
        if any(keyword in label_lower for keyword in ['loris', 'slow loris', 'slender loris']):
            return "Loris"
        if any(keyword in label_lower for keyword in ['galago', 'bushbaby', 'bush baby']):
            return "Galago"
        
        # Hyenas
        if 'hyena' in label_lower or 'hyaena' in label_lower:
            return "Hyena"
        
        # Reptiles - Snakes (specific)
        if any(keyword in label_lower for keyword in ['python', 'ball python', 'reticulated', 'burmese']):
            return "Python"
        if any(keyword in label_lower for keyword in ['cobra', 'king cobra', 'indian cobra', 'black cobra']):
            return "Cobra"
        if any(keyword in label_lower for keyword in ['mamba', 'green mamba', 'black mamba']):
            return "Mamba"
        if any(keyword in label_lower for keyword in ['boa', 'boa constrictor', 'anaconda']):
            return "Boa Constrictor"
        if any(keyword in label_lower for keyword in ['rattlesnake', 'diamondback', 'sidewinder']):
            return "Rattlesnake"
        if any(keyword in label_lower for keyword in ['viper', 'pit viper', 'horned viper']):
            return "Viper"
        if any(keyword in label_lower for keyword in ['sea snake', 'sea krait', 'krait']):
            return "Sea Snake"
        if any(keyword in label_lower for keyword in ['corn snake', 'milk snake', 'king snake', 'water snake', 'garter']):
            return "Snake"
        
        # Reptiles - Lizards (specific)
        if 'komodo' in label_lower:
            return "Komodo Dragon"
        if any(keyword in label_lower for keyword in ['monitor', 'monitor lizard', 'nile monitor']):
            return "Monitor Lizard"
        if any(keyword in label_lower for keyword in ['iguana', 'green iguana', 'common iguana']):
            return "Iguana"
        if any(keyword in label_lower for keyword in ['chameleon', 'veiled', 'panther', 'jackson', 'crested']):
            return "Chameleon"
        if any(keyword in label_lower for keyword in ['gecko', 'leopard gecko', 'day gecko', 'crested gecko']):
            return "Gecko"
        if 'frilled' in label_lower:
            return "Frilled Lizard"
        if 'gila monster' in label_lower:
            return "Gila Monster"
        if 'basilisk' in label_lower:
            return "Basilisk"
        
        # Reptiles - Turtles and Crocodilians
        if any(keyword in label_lower for keyword in ['tortoise', 'giant tortoise', 'galapagos', 'aldabra']):
            return "Tortoise"
        if any(keyword in label_lower for keyword in ['sea turtle', 'green sea turtle', 'loggerhead', 'hawksbill', 'leatherback', 'olive ridley']):
            return "Sea Turtle"
        if any(keyword in label_lower for keyword in ['turtle', 'box turtle', 'painted', 'slider', 'snapping']):
            return "Turtle"
        if any(keyword in label_lower for keyword in ['crocodile', 'nile', 'saltwater', 'saltie', 'american crocodile']):
            return "Crocodile"
        if any(keyword in label_lower for keyword in ['alligator', 'american alligator', 'gator']):
            return "Alligator"
        if any(keyword in label_lower for keyword in ['caiman', 'spectacled', 'black caiman']):
            return "Caiman"
        if any(keyword in label_lower for keyword in ['gharial', 'false gharial']):
            return "Gharial"
        
        # Amphibians (specific)
        if any(keyword in label_lower for keyword in ['poison dart', 'poison frog', 'dart frog']):
            return "Poison Dart Frog"
        if any(keyword in label_lower for keyword in ['tree frog', 'red-eyed']):
            return "Tree Frog"
        if any(keyword in label_lower for keyword in ['bullfrog', 'american bullfrog']):
            return "Bullfrog"
        if any(keyword in label_lower for keyword in ['frog', 'leopard frog', 'green frog', 'chorus', 'spring peeper']):
            return "Frog"
        if any(keyword in label_lower for keyword in ['toad', 'cane toad', 'bufo', 'true toad']):
            return "Toad"
        if 'axolotl' in label_lower:
            return "Axolotl"
        if any(keyword in label_lower for keyword in ['salamander', 'tiger salamander', 'spotted', 'fire salamander', 'european']):
            return "Salamander"
        if any(keyword in label_lower for keyword in ['newt', 'alpine', 'smooth', 'common newt']):
            return "Newt"
        
        # Marine mammals (specific)
        if any(keyword in label_lower for keyword in ['whale', 'blue whale', 'grey whale', 'gray whale', 'killer whale', 'orca', 'humpback']):
            return "Whale"
        if any(keyword in label_lower for keyword in ['dolphin', 'bottlenose', 'spinner']):
            return "Dolphin"
        if any(keyword in label_lower for keyword in ['seal', 'sea lion', 'fur seal', 'leopard seal']):
            return "Seal"
        if 'walrus' in label_lower:
            return "Walrus"
        if any(keyword in label_lower for keyword in ['otter', 'sea otter', 'river otter']):
            return "Otter"
        if any(keyword in label_lower for keyword in ['manatee', 'dugong', 'sea cow']):
            return "Manatee"
        if 'narwhal' in label_lower:
            return "Narwhal"
        if 'beluga' in label_lower:
            return "Beluga"
        
        # Ungulates/Hoofed animals (specific)
        if any(keyword in label_lower for keyword in ['zebra', 'plains', 'mountain zebra', 'grevy']):
            return "Zebra"
        if 'giraffe' in label_lower:
            return "Giraffe"
        if 'okapi' in label_lower:
            return "Okapi"
        if any(keyword in label_lower for keyword in ['hippopotamus', 'hippo']):
            return "Hippopotamus"
        if any(keyword in label_lower for keyword in ['rhinoceros', 'rhino', 'white rhino', 'black rhino', 'sumatran']):
            return "Rhinoceros"
        if any(keyword in label_lower for keyword in ['deer', 'red deer', 'mule deer', 'sika']):
            return "Deer"
        if 'elk' in label_lower or 'moose' in label_lower:
            return "Elk"
        if any(keyword in label_lower for keyword in ['reindeer', 'caribou']):
            return "Reindeer"
        if any(keyword in label_lower for keyword in ['antelope', 'pronghorn', 'gazelle', 'springbok', 'impala', 'kudu']):
            return "Antelope"
        if any(keyword in label_lower for keyword in ['eland', 'wildebeest', 'gnu', 'nyala', 'oryx', 'addax']):
            return "Wildebeest"
        if any(keyword in label_lower for keyword in ['gemsbuck', 'waterbuck', 'bushbuck', 'duiker']):
            return "African Ungulate"
        if any(keyword in label_lower for keyword in ['buffalo', 'water buffalo']):
            return "Buffalo"
        if any(keyword in label_lower for keyword in ['bison', 'american bison', 'european']):
            return "Bison"
        if any(keyword in label_lower for keyword in ['camel', 'dromedary', 'arabian', 'bactrian']):
            return "Camel"
        if any(keyword in label_lower for keyword in ['llama', 'alpaca', 'guanaco', 'vicuna']):
            return "Llama"
        if any(keyword in label_lower for keyword in ['sheep', 'lamb', 'ram', 'bighorn']):
            return "Sheep"
        if any(keyword in label_lower for keyword in ['goat', 'kid', 'mountain goat', 'ibex']):
            return "Goat"
        if any(keyword in label_lower for keyword in ['horse', 'pony', 'mare', 'stallion', 'colt', 'foal']):
            return "Horse"
        if any(keyword in label_lower for keyword in ['donkey', 'mule', 'ass']):
            return "Donkey"
        if any(keyword in label_lower for keyword in ['cattle', 'cow', 'bull', 'ox', 'holstein', 'hereford']):
            return "Cow"
        if any(keyword in label_lower for keyword in ['pig', 'hog', 'wild boar', 'babirusa', 'peccary', 'javelina']):
            return "Pig"
        if any(keyword in label_lower for keyword in ['yak', 'musk ox', 'takin']):
            return label_lower.title()
        
        # Small mammals
        if 'aardvark' in label_lower:
            return "Aardvark"
        if 'anteater' in label_lower or 'ant eater' in label_lower:
            return "Anteater"
        if 'pangolin' in label_lower:
            return "Pangolin"
        if 'hedgehog' in label_lower:
            return "Hedgehog"
        if any(keyword in label_lower for keyword in ['hyrax', 'rock hyrax', 'tree hyrax']):
            return "Hyrax"
        if any(keyword in label_lower for keyword in ['capybara', 'largest rodent']):
            return "Capybara"
        if 'paca' in label_lower or 'agouti' in label_lower:
            return label_lower.title()
        if 'quokka' in label_lower:
            return "Quokka"
        if 'numbat' in label_lower:
            return "Numbat"
        if 'tasmanian devil' in label_lower or 'tasmanian' in label_lower:
            return "Tasmanian Devil"
        if 'sugar glider' in label_lower or 'feathertail' in label_lower:
            return "Sugar Glider"
        if 'platypus' in label_lower:
            return "Platypus"
        if 'echidna' in label_lower:
            return "Echidna"
        if any(keyword in label_lower for keyword in ['tree shrew', 'tupaia']):
            return "Tree Shrew"
        if any(keyword in label_lower for keyword in ['colugo', 'flying lemur', 'flying dragon']):
            return "Colugo"
        if any(keyword in label_lower for keyword in ['emu', 'cassowary', 'kiwi', 'rhea']):
            return "Bird"
        
        # Other small mammals (keep specific names)
        if 'raccoon' in label_lower:
            return "Raccoon"
        if 'skunk' in label_lower:
            return "Skunk"
        if 'weasel' in label_lower:
            return "Weasel"
        if 'mink' in label_lower:
            return "Mink"
        if any(keyword in label_lower for keyword in ['badger', 'honey badger']):
            return "Badger"
        if any(keyword in label_lower for keyword in ['ferret', 'polecat', 'black-footed ferret']):
            return "Ferret"
        if any(keyword in label_lower for keyword in ['rabbit', 'hare']):
            return "Rabbit"
        if 'squirrel' in label_lower:
            return "Squirrel"
        if 'chipmunk' in label_lower:
            return "Chipmunk"
        if 'hamster' in label_lower:
            return "Hamster"
        if 'guinea pig' in label_lower:
            return "Guinea Pig"
        if 'mouse' in label_lower:
            return "Mouse"
        if 'rat' in label_lower:
            return "Rat"
        if 'beaver' in label_lower:
            return "Beaver"
        if 'porcupine' in label_lower:
            return "Porcupine"
        if 'armadillo' in label_lower:
            return "Armadillo"
        if 'sloth' in label_lower:
            return "Sloth"
        
        # Default: return capitalized version of the label
        return specific_label.title()

    def is_human(self, label):
        """Check if a label corresponds to a human"""
        label_lower = label.lower()
        human_keywords = [
            'person', 'human', 'man', 'woman', 'boy', 'girl', 'child',
            'baby', 'people', 'face', 'selfie', 'portrait', 'head', 'headshot',
            'baseball player', 'basketball player', 'soccer player', 'football player',
            'runner', 'athlete', 'climber', 'skier', 'snowboarder', 'swimmer',
            'bride', 'groom', 'cowboy', 'cowgirl', 'police officer', 'policeman', 'policewoman',
            'firefighter', 'doctor', 'nurse', 'chef', 'waiter', 'businessman', 'businesswoman',
            'student', 'scuba diver', 'soldier', 'surgeon'
        ]
        return any(keyword in label_lower for keyword in human_keywords)
    
    def is_animal(self, label):
        """Check if a label corresponds to an animal"""
        label_lower = label.lower()
        
        # Check all animal categories
        for category in ['dog', 'cat', 'bird', 'other_animal']:
            for keyword in self.animal_categories[category]:
                if keyword.lower() in label_lower:
                    return True
        
        # Additional animal keywords
        animal_keywords = [
            'animal', 'mammal', 'bird', 'fish', 'reptile', 'amphibian',
            'insect', 'arachnid', 'crustacean', 'mollusk', 'rodent',
            'carnivore', 'herbivore', 'omnivore', 'vertebrate', 'invertebrate'
        ]
        
        return any(keyword in label_lower for keyword in animal_keywords)

    def classify_and_organize(self, image_path):
        """
        Classify image and organize into ANIMAL categories
        Returns: (folder, detected_label, confidence)
        """
        try:
            # Open and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            batch_t = batch_t.to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(batch_t)
            
            # Get top 5 predictions
            probabilities = F.softmax(output[0], dim=0)
            top_probs, top_cat_ids = torch.topk(probabilities, 5)
            
            # Find the best animal or human match
            best_animal = None
            best_human = None
            best_confidence = 0
            best_label = ""
            
            for i in range(len(top_cat_ids)):
                label = self.all_labels[top_cat_ids[i].item()]
                confidence = top_probs[i].item()
                
                # Check if this is an animal
                if self.is_animal(label):
                    if confidence > best_confidence:
                        best_animal = label
                        best_confidence = confidence
                        best_label = label
                # Check if this is a human
                elif self.is_human(label):
                    if confidence > best_confidence:
                        best_human = label
                        best_confidence = confidence
                        best_label = label
            
            # If no animal or human detected in top 5, check all predictions
            if best_animal is None and best_human is None:
                # Get all predictions with confidence > 5% threshold (stricter)
                for idx, prob in enumerate(probabilities):
                    if prob > 0.05:  # 5% threshold - more strict
                        label = self.all_labels[idx]
                        if self.is_animal(label):
                            best_animal = label
                            best_confidence = prob.item()
                            best_label = label
                            break
                        elif self.is_human(label):
                            best_human = label
                            best_confidence = prob.item()
                            best_label = label
                            break
            
            # Determine category and GENERALIZE the label
            destination_folder = 'non_animal'
            detected_label = "Other"
            
            # Require minimum confidence for animal vs human (slightly lower for human to improve recall)
            MIN_CONFIDENCE_ANIMAL = 0.15
            MIN_CONFIDENCE_HUMAN = 0.10
            
            if best_animal and best_confidence >= MIN_CONFIDENCE_ANIMAL:
                # Map specific breed/species to general category
                general_label = self._generalize_animal_label(best_animal)
                detected_label = general_label
                
                # Check specific categories
                for category in ['dog', 'cat', 'bird']:
                    for keyword in self.animal_categories[category]:
                        if keyword.lower() in best_animal.lower():
                            destination_folder = category
                            break
                    if destination_folder != 'non_animal':
                        break
                
                # If not dog/cat/bird but still animal
                if destination_folder == 'non_animal':
                    destination_folder = 'other_animal'
            elif best_human and best_confidence >= MIN_CONFIDENCE_HUMAN:
                # Human detected with sufficient confidence
                detected_label = "Human"
                destination_folder = 'non_animal'
            else:
                # Not an animal or human - just label as "Other"
                best_confidence = top_probs[0].item()
                detected_label = "Other"
                print(f"  ⚠ No animal or human detected. Labeling as: Other ({best_confidence:.2%})")

            # Create gallery folder structure
            gallery_base = 'static/gallery'
            os.makedirs(gallery_base, exist_ok=True)
            
            for folder in ['dog', 'cat', 'bird', 'other_animal', 'non_animal']:
                os.makedirs(f'{gallery_base}/{folder}', exist_ok=True)

            # Move file to appropriate folder
            filename = os.path.basename(image_path)
            new_path = f'{gallery_base}/{destination_folder}/{filename}'
            shutil.move(image_path, new_path)
            
            # Save metadata
            self._save_metadata(new_path, detected_label, best_confidence, destination_folder)
            
            # Print results
            print(f"  Top 3 predictions:")
            for i in range(min(3, len(top_cat_ids))):
                label = self.all_labels[top_cat_ids[i].item()]
                prob = top_probs[i].item()
                animal_flag = "🐾" if self.is_animal(label) else "  "
                print(f"    {animal_flag} {label}: {prob:.2%}")

            return destination_folder, detected_label, best_confidence

        except Exception as e:
            print(f"Classification error: {e}")
            # Move to error folder as fallback
            os.makedirs('static/gallery/error', exist_ok=True)
            error_path = f'static/gallery/error/{os.path.basename(image_path)}'
            if os.path.exists(image_path):
                shutil.move(image_path, error_path)
            return 'error', str(e), 0.0

    def _save_metadata(self, image_path, label, confidence, category):
        """Save classification metadata as JSON"""
        import json
        from datetime import datetime
        
        metadata = {
            'filename': os.path.basename(image_path),
            'ai_label': label,
            'confidence': f"{confidence:.2%}",
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'is_animal': category != 'non_animal'
        }
        
        metadata_path = image_path.replace('.jpg', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# Test function
def test_classifier():
    """Test the animal classifier with sample images"""
    classifier = ImageClassifier()
    
    # Test with different types of images
    test_cases = [
        ("A dog photo", "dog"),
        ("A cat photo", "cat"),
        ("A bird photo", "bird"),
        ("A car (non-animal)", "non_animal")
    ]
    
    print("\n" + "="*60)
    print("ANIMAL CLASSIFIER TEST")
    print("="*60)
    
    for description, expected in test_cases:
        print(f"\nTesting: {description}")
        print(f"Expected: {expected}")
        # Note: You would need actual test images here
        print("(Add test images to test this functionality)")

if __name__ == "__main__":
    test_classifier()