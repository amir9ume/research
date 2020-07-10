#credits: Meghana Moorthy Bhat#


def build_bow_embeddings(doc_name, bow_w2ix, df_filtered):
    reviewer = dict()
    submitter = dict()
    for i in range(len(doc_name)):
        if "archive_papers" in doc_name[i]:
            author = doc_name[i].split('/')[-2]
            if author in df_filtered['reviewer'].values:
                if author not in reviewer:
                    reviewer[author] = torch.zeros(len(bow_w2ix))
                with open(doc_name[i], 'r') as f:
                    if ".bow" in doc_name[i]:
                        lines = f.readlines()
                        for line in lines:
                            w = line.split()[0]
                            count = int(line.split()[1])
                            if w in bow_w2ix:
                                reviewer[author][bow_w2ix[w]]+=count
        else:
            paper = doc_name[i].split('/')[-1]
            paper_id = re.sub('\D', '', paper)
            submitter[paper_id] = torch.zeros(len(bow_w2ix))
            with open(doc_name[i], 'r') as f:
                if ".bow" in doc_name[i]:
                    lines = f.readlines()
                    for line in lines:
                        words = line.split()
                        w = words[0]
                        count = int(words[1])
                        if w in bow_w2ix:
                            submitter[paper_id][bow_w2ix[w]]=count

    return reviewer, submitter

