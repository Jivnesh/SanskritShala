from django import template

register = template.Library()

@register.filter(name='sub_range' ,is_safe=True)
def sub_range(value, arg):
    return range(value-arg)

@register.filter(name='addvalue' ,is_safe=True)
def addvalue(value, arg):
    return value+arg

@register.filter(name='subvalue')
def subvalue(value, arg):
    return value-arg

@register.filter(name='getdicvalue' ,is_safe=True)
def getdicvalue(value, arg):
    return value[arg]

@register.filter(name='updatevale' ,is_safe=True)
def updatevale(value, arg):
	value = value+arg
	return value

@register.filter(name='checkpos' ,is_safe=True)
def checkpos(value, arg):
	check = 'no'
	for val in arg :
		if val == value :
			check = 'ok'
			break
	return check

@register.filter(name='getword' ,is_safe=True)
def getword(value, args):
	v = args.split('-')
	l = int(v[0])
	p = int(v[1])
	for wd in value :
		if wd.level == l and wd.position == p :
			return wd
	return 'error'

@register.filter(name='getallwordids' ,is_safe=True)
def getallwordids(value, args):
	v = args.split('-')
	l = int(v[0])
	p = int(v[1])
	wdids = ''
	for wd in value :
		if wd.level == l and wd.position == p :
			if wdids == '' :
				wdids = str(wd.id)
			else :
				wdids = wdids+'-'+str(wd.id)
	return wdids


@register.filter(name='getstring',is_safe=True)
def getstring(value, arg):
	s = str(value)+'-'+str(arg)
	return s

@register.filter(name='getwordmorphdata',is_safe=True)
def getwordmorphdata(value, arg):
	for wds in value :
		if wds.id == int(arg):
			wd = wds;
			break
	lemma = wd.lemma
	if not str(wd.pre_verb) == '' :
		lemma = wd.pre_verb+'-'+lemma
	if not str(wd.aux_info) == '' :
		if wd.aux_info[-18:-1]=='sence of lemma = ' :
			lemma = lemma+'-'+wd.aux_info[-1:]
			if not str(wd.aux_info[:-18]) == ' ' :
				lemma = lemma +' ('+wd.aux_info[:-18]+')'
		else :
			lemma = lemma +' ('+ wd.aux_info+')'

	s = '{ '+wd.morph+' }@['+lemma + ']' 
	return s


@register.filter(name='nbios',is_safe=True)
def nbios(value, arg):
	s = value.split('-')

	return len(s)

@register.filter(name='nbinputs',is_safe=True)
def nbinputs(value, arg):
	s = arg.split('-')
	inps = []

	for i in s :
		inps.append(getwordmorphdata(value,i))
	return "\n".join(inps)
