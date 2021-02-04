import pygame
import random
import math
import numpy as np
from pygame import gfxdraw

from Protein import Protein
from UIcontrols import Settings


def main():
	showCreationBots = False
	showBotsAllActivities = False
	min_bot_count = 10
	starting_Bot_Count = 20
	poison_ratio = 0.05
	food_ratio = 0.2
	mutation_rate = 0.005
	steering_weights = 0.2
	dna_length = 1000
	perception_radius_mutation_range = 10
	#reproduction_rate = 0.0005
	reproduction_rate = 0.0008
	initial_perception_radius = 1
	boundary_size = 10
	max_vel = 10
	initial_max_force = 0.02
	initial_health = 100
	max_poison = 25
	nutrition = [20, -80]
	organisms = []
	food = []
	poison = []
	proteins = []
	oldest_ever = 0
	oldest_ever_dna = []
	start_codon = 'ATG'
	stop_codons = ['TAA', 'TAG', 'TGA']


	aminoAcid_dictionary = {"TTT": "F", "CTT": "L", "ATT": "I", "GTT": "V",
							"TTC": "F", "CTC": "L", "ATC": "I", "GTC": "V",
							"TTA": "L", "CTA": "L", "ATA": "I", "GTA": "V",
							"TTG": "L", "CTG": "L", "ATG": "M", "GTG": "V",
							"TCT": "S", "CCT": "P", "ACT": "T", "GCT": "A",
							"TCC": "S", "CCC": "P", "ACC": "T", "GCC": "A",
							"TCA": "S", "CCA": "P", "ACA": "T", "GCA": "A",
							"TCG": "S", "CCG": "P", "ACG": "T", "GCG": "A",
							"TAT": "Y", "CAT": "H", "AAT": "N", "GAT": "D",
							"TAC": "Y", "CAC": "H", "AAC": "N", "GAC": "D",
							"TAA": "&", "CAA": "Q", "AAA": "K", "GAA": "E",
							"TAG": "_", "CAG": "Q", "AAG": "K", "GAG": "E",
							"TGT": "C", "CGT": "R", "AGT": "S", "GGT": "G",
							"TGC": "C", "CGC": "R", "AGC": "S", "GGC": "G",
							"TGA": "_", "CGA": "R", "AGA": "R", "GGA": "G",
							"TGG": "W", "CGG": "R", "AGG": "R", "GGG": "G"
							}



	# Pygame Options
	pygame.init()
	game_width = 800
	game_height = 400
	black = (0, 0, 0)
	red = (255, 0, 0)
	blue = (65, 131, 196)
	green = (0, 255, 0)
	fps = 60

	gameDisplay = pygame.display.set_mode((game_width, game_height))
	clock = pygame.time.Clock()



	def make_proteins():
		proteins.append(Protein("food_perception", "ST"))
		proteins.append(Protein("poison_perception", "AI"))
		proteins.append(Protein("red_tail+", "QR"))
		proteins.append(Protein("red_tail-", "RR"))
		proteins.append(Protein("green_tail+", "PR"))
		proteins.append(Protein("green_tail-", "LR"))



	# NI ==> AATATT
	# QP ==> CAGACG

	def lerp():
		percent_health = organism.health/settings.initial_health
		lerped_colour = (max(min((1-percent_health)*255,255),0), max(min(percent_health*255,255),0), 0)
		return lerped_colour

	def magnitude_calc(vector):
		x = 0
		for i in vector:
			x += i**2
		magnitude = x**0.5
		return(magnitude)

	def normalise(vector):
		magnitude = magnitude_calc(vector)
		if magnitude != 0:
			vector = vector/magnitude
		return(vector)



	class create_dna():
		def __init__(self, base_dna=None):
			self.genetic_code = None
			self.protein_sequence = None
			self.proteins = []

			if base_dna == None:
				#Init
				#self.genetic_code = initial_genetic_code
				self.genetic_code = self.generateDNASequence(settings.dna_length)
			else:
				#Clone
				self.genetic_code = base_dna.genetic_code
				self.mutate_genetic_code()

			self.protein_sequence = self.getAminoAcidSequence()
			self.find_proteins()

		def generateDNASequence(self, length):
			l = ['C', 'A', 'G', 'T']
			res = []
			for i in range(0, length):
				res.append(random.choice(l))
			return ''.join([str(e) for e in res])

		def alter_nucleotide(self, random_rate):
			l = ['C', 'A', 'G', 'T']
			if random.random() < random_rate:
				changepos = random.randint(0, len(self.genetic_code) - 1)
				dl = np.array(list(self.genetic_code))
				ch = "" + dl[changepos]
				l.remove(ch)
				ms = random.choice(l)
				dl[changepos] = ms
				self.genetic_code = ''.join([str(e) for e in dl])


		def add_nucleotide(self, random_rate):
			l = ['C', 'A', 'G', 'T']
			if random.random() < random_rate:
				changepos = random.randint(0, len(self.genetic_code) - 1)
				dl = np.array(list(self.genetic_code))
				ms = random.choice(l)
				dl = np.insert(dl, changepos, ms)
				self.genetic_code = ''.join([str(e) for e in dl])

		def remove_nucleotide(self, random_rate):
			if random.random() < random_rate:
				changepos = random.randint(0, len(self.genetic_code) - 1)
				dl = np.array(list(self.genetic_code))
				dl = np.delete(dl, changepos)
				self.genetic_code = ''.join([str(e) for e in dl])

		def multiple_nucleotide(self, random_rate):
			if random.random() < random_rate:
				self.genetic_code += self.genetic_code

		def mutate_genetic_code(self):
			count = int(dna_length * mutation_rate)
			for i in range(count):
				self.alter_nucleotide(0.4)
				self.add_nucleotide(0.4)
				self.remove_nucleotide(0.1)
				self.multiple_nucleotide(0.002)

		def protein_count(self, protein_name):
			count = 0
			for protein in self.proteins:
				if protein_name == protein.name:
					count += 1
			return count

		def find_proteins(self):
			for protein in proteins:
				protein_count = self.protein_sequence.count(protein.sequence)
				for i in range(protein_count):
					self.proteins.append(protein)

		def replicate(self):
			dna = create_dna(self)
			return dna

		def show(self, msg):
			print(msg, ',greenForce:', self.green_force, 'redForce:', self.red_force, "seq:", self.protein_sequence)
				  #'foodPerception:',
				  #self.food_perception, 'poisonPerception:',
				  #self.poison_perception)


		def getAminoAcidSequence(self):
			sequence = ""
			divisions = []
			len_code = len(self.genetic_code)
			start = self.genetic_code.find(start_codon)
			min_stop = 0
			while start >= 0 and min_stop >= 0:
				min_stop = -1
				stop = -1
				for stop_codon in stop_codons:
					stop = self.genetic_code.find(stop_codon, start + 3, len_code)
					if min_stop == -1 or stop < min_stop:
						min_stop = stop
				if min_stop >= 0:
					divisions.append(self.genetic_code[start + 3:min_stop])
					start = self.genetic_code.find(start_codon, min_stop + 3, len_code)

			for division in divisions:
				if division is None or division == "":
					continue
				sequence += '&'
				x = len(division) - (3 + len(division) % 3)
				for i in range(0, x + 1, 3):
					# if aminoAcid_dictionary[self.code[i:i + 3]] == "_":
					#    break
					sequence += aminoAcid_dictionary[division[i:i + 3]]
				sequence += '_'
			return sequence

	class create_organism():
		def __init__(self, x, y, replicating_dna=None):
			# Genetic Features
			self.food_perception = None
			self.poison_perception = None
			self.red_force = initial_max_force
			self.green_force = initial_max_force
			#
			self.position = np.array([x,y], dtype='float64')
			self.velocity = np.array([random.uniform(-settings.max_vel,settings.max_vel), random.uniform(-settings.max_vel, settings.max_vel)], dtype='float64')
			self.acceleration = np.array([0, 0], dtype='float64')
			self.colour = green
			self.health = settings.initial_health
			self.max_vel = 2
			self.max_force = 0.5
			self.size = 5
			self.age = 1
			self.dna = None

			# DNA Replication
			if replicating_dna != None:
				# Mutation
				self.dna = replicating_dna.replicate()
				msg = "Child "
			# New DNA
			else:
				self.dna = create_dna()
				msg = "New   "
			#
			self.set_features_from_proteins(replicating_dna)
			if showCreationBots:
				self.show(msg)

		def set_features_from_proteins(self, replicating_dna):
			self.food_perception = 10 * self.dna.protein_count("food_perception") + random.uniform(0, initial_perception_radius)
			self.poison_perception = 10 * self.dna.protein_count("poison_perception") + random.uniform(0, initial_perception_radius)
			self.green_force = 0
			self.red_force = 0
			self.green_force += settings.steering_weights * self.dna.protein_count("green_tail+") + random.uniform(0, initial_max_force)
			self.green_force += -settings.steering_weights * self.dna.protein_count("green_tail-") + random.uniform(0, -initial_max_force)
			self.red_force += settings.steering_weights * self.dna.protein_count("red_tail+") + random.uniform(0, initial_max_force)
			self.red_force += -settings.steering_weights * self.dna.protein_count("red_tail-") + random.uniform(0, -initial_max_force)

		def update(self):
			self.velocity += self.acceleration
			self.velocity = normalise(self.velocity)*self.max_vel
			self.position += self.velocity
			self.acceleration *= 0
			self.health -= 0.2
			self.colour = lerp()
			self.health = min(settings.initial_health, self.health)
			self.age += 1


		def show(self, msg):
			print(msg,
				  'foodPerception:', round(self.food_perception, 2), 'poisonPerception:',
				  round(self.poison_perception, 2), "seq:", self.dna.protein_sequence, )

		def reproduce(self):
			if random.random() < settings.reproduction_rate:
				if showCreationBots:
					self.show("Parent")
				organisms.append(create_organism(self.position[0], self.position[1], self.dna))


		def dead(self):
			if self.health > 0:
				return(False)
			else:
				if self.position[0] < game_width - boundary_size and self.position[0] > boundary_size and self.position[1] < game_height - boundary_size and self.position[1] > boundary_size:
					food.append(self.position)
				return(True)

		def apply_force(self, force):
			self.acceleration += force

		def seek(self, target):
			desired_vel = np.add(target, -self.position)
			desired_vel = normalise(desired_vel)*self.max_vel
			steering_force = np.add(desired_vel, -self.velocity)
			steering_force = normalise(steering_force)*self.max_force
			return(steering_force)

		def eat(self, list_of_stuff, index):
			closest = None
			closest_distance = max(game_width, game_height)
			bot_x = self.position[0]
			bot_y = self.position[1]
			item_number = len(list_of_stuff)-1
			for i in list_of_stuff[::-1]:
				item_x = i[0]
				item_y = i[1]
				distance = math.hypot(bot_x-item_x, bot_y-item_y)
				if distance < 5:
					list_of_stuff.pop(item_number)
					self.health += nutrition[index]
				if distance < closest_distance:
					closest_distance = distance
					closest = i
				item_number -=1
			if index == 0:
				perception = self.food_perception
				force = self.green_force
			else:
				perception = self.poison_perception
				force = self.red_force
			if closest_distance < perception:
				seek = self.seek(closest) # index)
				seek *= force
				seek = normalise(seek)*self.max_force
				self.apply_force(seek)


		def boundaries(self):
			desired = None
			x_pos = self.position[0]
			y_pos = self.position[1]
			if x_pos < boundary_size:
				desired = np.array([self.max_vel, self.velocity[1]])
				steer = desired-self.velocity
				steer = normalise(steer)*self.max_force
				self.apply_force(steer)
			elif x_pos > game_width - boundary_size:
				desired = np.array([-self.max_vel, self.velocity[1]])
				steer = desired-self.velocity
				steer = normalise(steer)*self.max_force
				self.apply_force(steer)
			if y_pos < boundary_size:
				desired = np.array([self.velocity[0], self.max_vel])
				steer = desired-self.velocity
				steer = normalise(steer)*self.max_force
				self.apply_force(steer)
			elif y_pos > game_height - boundary_size:
				desired = np.array([self.velocity[0], -self.max_vel])
				steer = desired-self.velocity
				steer = normalise(steer)*self.max_force
				self.apply_force(steer)
			'''if desired != None:
				steer = desired-self.velocity
				steer = normalise(steer)*self.max_force
				self.apply_force(steer)'''

		def draw_bot(self, index):
			pygame.gfxdraw.aacircle(gameDisplay, int(self.position[0]), int(self.position[1]), 10, self.colour)
			filled_color = self.colour
			if settings.current_selection == index:
				filled_color = blue
			pygame.gfxdraw.filled_circle(gameDisplay, int(self.position[0]), int(self.position[1]), 10, filled_color)
			pygame.draw.circle(gameDisplay, green, (int(self.position[0]), int(self.position[1])),
							   abs(int(self.food_perception)), abs(int(min(2, self.food_perception))))
			pygame.draw.circle(gameDisplay, red, (int(self.position[0]), int(self.position[1])), abs(int(self.poison_perception)),
							   abs(int(min(2, self.poison_perception))))
			pygame.draw.line(gameDisplay, green, (int(self.position[0]), int(self.position[1])), (
				int(self.position[0] + (self.velocity[0] * self.green_force * 25)),
				int(self.position[1] + (self.velocity[1] * self.green_force * 25))), 3)
			pygame.draw.line(gameDisplay, red, (int(self.position[0]), int(self.position[1])), (
				int(self.position[0] + (self.velocity[0] * self.red_force * 25)),
				int(self.position[1] + (self.velocity[1] * self.red_force * 25))), 2)

	# Starting
	make_proteins()
	# Settings UI
	settings = Settings(proteins)
	settings.poison_rate = poison_ratio
	settings.food_rate = food_ratio
	settings.min_bot_count = min_bot_count
	settings.max_vel = max_vel
	settings.initial_health = initial_health
	settings.max_poison = max_poison
	settings.mutation_rate = mutation_rate
	settings.reproduction_rate = reproduction_rate
	settings.steering_weights = steering_weights
	settings.dna_length = dna_length
	#
	print(create_dna().protein_sequence)
	for i in range(starting_Bot_Count):
		organisms.append(create_organism(random.uniform(0,game_width),random.uniform(0,game_height)))
	running = True
	# Events
	every_second_event, t, trail = pygame.USEREVENT + 1, 500, []
	pygame.time.set_timer(every_second_event, t)
	proteins_event, t2, trail2 = pygame.USEREVENT + 2, 5000, []
	pygame.time.set_timer(proteins_event, t2)
	listbox_event, t3, trail3 = pygame.USEREVENT + 3, 2000, []
	pygame.time.set_timer(listbox_event, t3)
	# LOOP
	while(running):
		settings.root.update()
		gameDisplay.fill(black)
		if len(organisms)< settings.min_bot_count or random.random() < 0.0001:
			organisms.append(create_organism(random.uniform(0,game_width),random.uniform(0,game_height)))
		if random.random()<settings.food_rate:
			food.append(np.array([random.uniform(boundary_size, game_width-boundary_size), random.uniform(boundary_size, game_height-boundary_size)], dtype='float64'))
		if random.random()<settings.poison_rate:
			poison.append(np.array([random.uniform(boundary_size, game_width-boundary_size), random.uniform(boundary_size, game_height-boundary_size)], dtype='float64'))
		if len(poison)>settings.max_poison:
			poison.pop(0)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			#print(event)
			elif event.type == every_second_event:
				settings.root.title('Number of Organisms: ' + str(len(organisms)))
			elif event.type == proteins_event:
				proteins = []
				proteins = settings.user_proteins
			elif event.type == listbox_event:
				create_organism_list(organisms, settings)

		index = -1
		for organism in organisms[::-1]:
			index += 1
			organism.eat(food, 0)
			organism.eat(poison, 1)
			organism.boundaries()
			organism.seek(pygame.mouse.get_pos())
			organism.update()
			if organism.age > oldest_ever:
				oldest_ever = organism.age
				oldest_ever_dna = organism.dna
				if showBotsAllActivities:
					print(oldest_ever, oldest_ever_dna)
			organism.draw_bot(index)
			if organism.dead():
				organisms.remove(organism)
			else:
				organism.reproduce()


		for i in food:
			pygame.draw.circle(gameDisplay, (0, 255, 0), (int(i[0]), int(i[1])), 3)
		for i in poison:
			pygame.draw.circle(gameDisplay, (255, 0, 0), (int(i[0]), int(i[1])), 3)
		pygame.display.update()
		clock.tick(fps)

	pygame.quit()
	quit()


def create_organism_list(organisms, settings):
	entry_list = []
	ind = 0
	for org in organisms:
		ind += 1
		str_protein = 'Proteins: None '
		for protein in org.dna.proteins:
			str_protein = str_protein.replace("None", "")
			str_protein += protein.sequence + '-'
		entry = str(ind) + ') ' + str_protein + '    Seq: ' + org.dna.protein_sequence
		entry_list.append(entry)
	settings.listbox_widget = entry_list


main()
