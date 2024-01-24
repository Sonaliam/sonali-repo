import copy

from collections import defaultdict, OrderedDict
import numpy as np
from sortedcontainers import SortedListWithKey

class AbstractBeam():
    def __init__(self, size):
        # note: here we assume bigger scores are better
        self.hypotheses = SortedListWithKey(key=lambda x: -x)
        self.size = size

    def add(self, hyp):
        self.hypotheses.add(hyp)
        if len(self.hypotheses) > self.size:
            assert len(self.hypotheses) == self.size + 1
            del self.hypotheses[-1]

    def __len__(self):
        return len(self.hypotheses)

    def __iter__(self):
        for hyp in self.hypotheses:
            yield hyp

# Thinking buffer
def init_coverage(constraints):
    coverage = []
    for c in constraints:
        coverage.append(np.zeros(len(c), dtype='int16'))
    return coverage



    class Con_Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=38, beam_implementation=AbstractBeam):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout)for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(hid_dim)
        self.l1 = nn.Linear(hid_dim*3, 1)
        self.l2 = nn.Linear(hid_dim * 3, 1)
        self.embed = nn.Embedding(vocab_len, 256) #embed_size=256
        self.beam_implementation = beam_implementation

    def forward(self, trg, enc_art, trg_mask, art_mask, imgs, enc_ent, enc_ient, ients, mode):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        trg = self.tok_embedding(trg)
        trg = trg * self.scale
        trg = self.dropout(trg + self.pos_embedding(pos))

        trg_decoder_out = trg
      
        for layer in self.layers:
            trg_decoder_out, attention = layer(trg_decoder_out, enc_art, trg_mask, art_mask, imgs, enc_ent, enc_ient)
      
        output = self.fc_out(trg_decoder_out)

        if mode == 'val':
            t = output.shape[1]
            START_TOKEN = vocab1.idx2word[0]
            P_START = 1.0
            N_BEST = 2
            entity_constraints = ients
            grid_height = entity_constraints.shape[1] 

            search_grid = OrderedDict()

            constraint_tm = constraintTranslationModel(vocab1)
            con_method = ConstrainedMethod(hyp_generation_func=constraint_tm.constraint_generate,
                                        constraint_generation_func=constraint_tm.constraint_generate_from_constraints,
                                        continue_constraint_func=constraint_tm.constraint_continue_unfinished_constraint,
                                        beam_implementation=AbstractBeam)
            start_hyp = ConstraintHypothesis(token=START_TOKEN, #score=P_START,
                                            coverage=init_coverage(entity_constraints),
                                            constraints=entity_constraints,
                                            payload=None,
                                            backpointer=None,
                                            unfinished_constraint=False
                                            )

            start_beam = self.beam_implementation(size=1)
            start_beam.add(start_hyp)

            search_grid[(0,0)] = start_beam

            output_grid = con_method.search(start_hyp=start_hyp, constraints=entity_constraints, max_source_len=trg_len, beam_size=10)

        return output, attention

class ConstrainedMethod(object):

    def __init__(self, hyp_generation_func, constraint_generation_func, continue_constraint_func,
                 beam_implementation=AbstractBeam):
        self.hyp_generation_func = hyp_generation_func
        self.constraint_generation_func = constraint_generation_func
        self.continue_constraint_func = continue_constraint_func
        self.beam_implementation = beam_implementation


    # QUESTION: are mid-constraint hyps allowed to fall off of the beam or not?
    def search(self, start_hyp, constraints, max_source_len, beam_size=10):


        grid_height = sum(len(c) for c in constraints)

        search_grid = OrderedDict()

        # a beam with one hyp starts the search
        start_beam = self.beam_implementation(size=1)
        start_beam.add(start_hyp)

        search_grid[(0,0)] = start_beam

        for i in range(1, max_source_len + 1):
            j_start = max(i - (max_source_len - grid_height), 0)
            j_end = min(i, grid_height) + 1
            beams_in_i = j_end - j_start
            for j in range(j_start, min(i, grid_height) + 1):
                # create the new beam
                new_beam = self.beam_implementation(size=beam_size)
                if (i-1, j) in search_grid:
                    generation_hyps = self.get_generation_hyps(search_grid[(i-1, j)])
                    for hyp in generation_hyps:
                        new_beam.add(hyp)
                # lower left diagonal cell adds hyps from constraints
                if (i-1, j-1) in search_grid:
                    new_constraint_hyps = self.get_new_constraint_hyps(search_grid[(i-1, j-1)])
                    continued_constraint_hyps = self.get_continued_constraint_hyps(search_grid[(i-1, j-1)])
                    new_beam.add(hyp)
                    for hyp in continued_constraint_hyps:
                        new_beam.add(hyp)

                search_grid[(i,j)] = new_beam

        return search_grid

    def get_generation_hyps(self, beam, output):
        """return all hyps which are continuations of the hyps on this beam

        hyp_generation_func maps `(hyp) --> continuations`

        the coverage vector of the parent hyp is not modified in each child
        """

        continuations = (self.hyp_generation_func(hyp) for hyp in beam if not hyp.unfinished_constraint)

        # flatten
        return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

    def get_new_constraint_hyps(self, beam):
        """return all hyps which start a new constraint from the hyps on this beam

        constraint_hyp_func maps `(hyp) --> continuations`

        the coverage vector of the parent hyp is modified in each child
        """

        continuations = (self.constraint_generation_func(hyp)
                         for hyp in beam if not hyp.unfinished_constraint)

        # flatten
        return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)


    def get_continued_constraint_hyps(self, beam):
        """return all hyps which continue the unfinished constraints on this beam

        constraint_hyp_func maps `(hyp, constraints) --> forced_continuations`

        the coverage vector of the parent hyp is modified in each child

        """
        continuations = (self.continue_constraint_func(hyp)
                         for hyp in beam if hyp.unfinished_constraint)

        return continuations



class ConstraintHypothesis:

    def __init__(self, token, coverage, constraints, payload=None, backpointer=None,
                 constraint_index=None, unfinished_constraint=False): #score,
        self.token = token

        assert len(coverage) == len(constraints), 'constraints and coverage length must match'
        assert all(len(cov) == len(cons) for cov, cons in zip(coverage, constraints)), \
            'each coverage and constraint vector must match'

        self.coverage = coverage
        self.constraints = constraints
        self.backpointer = backpointer
        self.payload = payload
        self.constraint_index = constraint_index
        self.unfinished_constraint = unfinished_constraint

    def __str__(self):
        return u'token: {}, sequence: {}, coverage: {}, constraints: {},'.format(
            self.token, self.sequence, self.coverage, self.constraints)

    def __getitem__(self):
        return getattr(self)

    @property
    def sequence(self):
        sequence = []
        current_hyp = self
        while current_hyp.backpointer is not None:
            sequence.append((current_hyp.token, current_hyp.constraint_index))
            current_hyp = current_hyp.backpointer
        sequence.append((current_hyp.token, current_hyp.constraint_index))
       return sequence[::-1]

    def constraint_candidates(self):
        available_constraints = []
        for idx in range(len(self.coverage)):
            if self.coverage[idx][0] == 0:
                available_constraints.append(idx)

        return available_constraints


class constraintTranslationModel(object):

    def __init__(self, vocab1):
        self.vocabulary = vocab1

    def constraint_generate(self, hyp, output, n_best=1):

        new_hyps = []
        for i in range(n_best):
            new_hyp = ConstraintHypothesis(token=next_tokens[i],
                                          #  score=next_scores[i],
                                           coverage=copy.deepcopy(hyp.coverage),
                                           constraints=hyp.constraints,
                                           payload=None,
                                           backpointer=hyp,
                                           constraint_index=None,
                                           unfinished_constraint=False
                                          )
            new_hyps.append(new_hyp)

        return new_hyps

    def constraint_generate_from_constraints(self, hyp):
        """Look at the coverage of the hyp to get constraint candidates"""

        assert hyp.unfinished_constraint is not True, 'hyp must not be part of an unfinished constraint'
        new_constraint_hyps = []
        available_constraints = hyp.constraint_candidates()
        for idx in available_constraints:
            # starting a new constraint
            constraint_token = hyp.constraints[idx][0]
            coverage = copy.deepcopy(hyp.coverage)
            coverage[idx][0] = 1
            if len(coverage[idx]) > 1:
                unfinished_constraint = True
            else:
                unfinished_constraint = False

            new_hyp = ConstraintHypothesis(token=constraint_token,
                                          #  score=score,
                                           coverage=coverage,
                                           constraints=hyp.constraints,
                                           payload=None,
                                           backpointer=hyp,
                                           constraint_index=(idx, 0),
                                           unfinished_constraint=unfinished_constraint
                                          )
            new_constraint_hyps.append(new_hyp)

        return new_constraint_hyps


    def constraint_continue_unfinished_constraint(self, hyp):
        assert hyp.unfinished_constraint is True, 'hyp must be part of an unfinished constraint'

        constraint_row_index = hyp.constraint_index[0]
        # the index of the next token in the constraint
        constraint_tok_index = hyp.constraint_index[1] + 1
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token = hyp.constraints[constraint_index[0]][constraint_index[1]]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1

        if len(hyp.constraints[constraint_row_index]) > constraint_tok_index + 1:
            unfinished_constraint = True
        else:
            unfinished_constraint = False

        new_hyp = ConstraintHypothesis(token=continued_constraint_token,
                                       coverage=coverage,
                                       constraints=hyp.constraints,
                                       payload=None,
                                       backpointer=hyp,
                                       constraint_index=constraint_index,
                                       unfinished_constraint=unfinished_constraint
                                      )
        return new_hyp        