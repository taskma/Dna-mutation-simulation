from tkinter import *
from Protein import Protein

class Settings():
    root = Tk(className="Control Panel" )
    root.geometry("1200x300")

    def listbox_widget_event(self, event):
        value = self._listbox_widget.curselection()[0]
        self.current_selection = value

    def __init__(self, proteins):
        self._proteins = proteins
        self.current_selection = -1

        self._var = DoubleVar()
        self._poison_rate = Scale(self.root, variable=self._var, label='Poison Rate', orient=HORIZONTAL, length=100)
        self._poison_rate.place(y=5, x=5)

        self._var2 = DoubleVar()
        self._food_rate = Scale(self.root, variable=self._var2, label='Food Rate', orient=HORIZONTAL, length=100)
        self._food_rate.place(y=5, x=120)

        self._var3 = DoubleVar()
        self._min_bot_count = Scale(self.root, variable=self._var3, label='Min Organism Count', orient=HORIZONTAL, length=135)
        self._min_bot_count.place(y=5, x=230)

        self._var4 = DoubleVar()
        self._mutation_rate = Scale(self.root, variable=self._var4, label='Mutation Rate', orient=HORIZONTAL, length=100, from_=0, to=1000)
        self._mutation_rate.place(y=5, x=375)

        self._var6 = DoubleVar()
        self._steering_weights = Scale(self.root, variable=self._var6, label='Steering_Weights', orient=HORIZONTAL, length=125)
        self._steering_weights.place(y=5, x=485)

        self._var7 = DoubleVar()
        self._reproduction_rate = Scale(self.root, variable=self._var7, label='Reproduction Rate', orient=HORIZONTAL, length=125, from_=0, to=1000 )
        self._reproduction_rate.place(y=5, x=615)

        self._var8= DoubleVar()
        self._max_vel = Scale(self.root, variable=self._var8, label='Max Velocity', orient=HORIZONTAL, length=100)
        self._max_vel.place(y=5, x=745)

        self._var9 = DoubleVar()
        self._dna_length = Scale(self.root, variable=self._var9, label='Dna Length', orient=HORIZONTAL, length=100, from_=100, to=10000)
        self._dna_length.place(y=5, x=855)

        self._var10 = DoubleVar()
        self._initial_health = Scale(self.root, variable=self._var10, label='Initial Health', orient=HORIZONTAL, length=100, from_=0, to=200)
        self._initial_health.place(y=5, x=965)

        self._var11 = DoubleVar()
        self._max_poison = Scale(self.root, variable=self._var11, label='Max Poison', orient=HORIZONTAL, length=100, from_=0, to=200)
        self._max_poison.place(y=5, x=1075)

        lab = Label(self.root, text="Protein Name   -            Sequence")
        lab.place(y=70, x=5)

        self._protein_names = []
        self._protein_sequences = []

        for i in range(len(proteins)):
            _protein_name = Entry(self.root,  width=15)
            _protein_name.insert(INSERT, self._proteins[i].name)
            _protein_name.place(y=100 + (i * 20), x=5)

            _protein_sequence = Entry(self.root, width=22)
            _protein_sequence.insert(INSERT, self._proteins[i].sequence)
            _protein_sequence.place(y=100 + (i * 20), x=150)

            self._protein_names.append(_protein_name)
            self._protein_sequences.append(_protein_sequence)

        self._listbox_widget = Listbox(self.root)
        self._listbox_widget.bind('<Double-1>', self.listbox_widget_event)
        self._listbox_widget.place(y=80, x=360, width=800, height=200)



    @property
    def user_proteins(self):
        proteins = []
        for i in range(len(self._protein_names)):
            name = self._protein_names[i].get()
            sequence = self._protein_sequences[i].get()
            proteins.append(Protein(name, sequence))
        return proteins

    @property
    def listbox_widget(self):
        return self._listbox_widget

    @listbox_widget.setter
    def listbox_widget(self, entry_list):
        self._listbox_widget.delete(0, 'end')
        for entry in entry_list:
            self._listbox_widget.insert(END, entry)
        if self.current_selection < self._listbox_widget.size():
            self._listbox_widget.selection_set(first=self.current_selection)
        else:
            self.current_selection = -1

    @property
    def poison_rate(self):
        return self._poison_rate.get() / 100

    @poison_rate.setter
    def poison_rate(self, val):
        self._poison_rate.set(val * 100)

    @property
    def food_rate(self):
        return self._food_rate.get() / 100

    @food_rate.setter
    def food_rate(self, val):
        self._food_rate.set(val * 100)

    @property
    def min_bot_count(self):
        return self._min_bot_count.get()

    @min_bot_count.setter
    def min_bot_count(self, val):
        self._min_bot_count.set(val)


    @property
    def mutation_rate(self):
        return self._mutation_rate.get() / 1000

    @mutation_rate.setter
    def mutation_rate(self, val):
        self._mutation_rate.set(val * 1000)

    @property
    def steering_weights(self):
        return self._steering_weights.get() / 100

    @steering_weights.setter
    def steering_weights(self, val):
        self._steering_weights.set(val * 100)


    @property
    def reproduction_rate(self):
        return self._reproduction_rate.get() / 10000

    @reproduction_rate.setter
    def reproduction_rate(self, val):
        self._reproduction_rate.set(val * 10000)

    @property
    def max_vel(self):
        return self._max_vel.get()

    @max_vel.setter
    def max_vel(self, val):
        self._max_vel.set(val)

    @property
    def dna_length(self):
        return self._dna_length.get()

    @dna_length.setter
    def dna_length(self, val):
        self._dna_length.set(val)

    @property
    def initial_health(self):
        return self._initial_health.get()

    @initial_health.setter
    def initial_health(self, val):
        self._initial_health.set(val)

    @property
    def max_poison(self):
        return self._max_poison.get()

    @max_poison.setter
    def max_poison(self, val):
        self._max_poison.set(val)


