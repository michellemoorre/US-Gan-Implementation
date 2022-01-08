from tqdm import tqdm

def train_batch(model, X_batch, y_batch, gan_optimizer, loss_function):
    X_batch = X_batch.to(model.device)
    y_batch = y_batch.to(model.device)
    
    opt_generator, opt_discriminator = gan_optimizer
    opt_discriminator.zero_grad()
    discriminator_loss = model.discriminator_loss(X_batch)
    discriminator_loss.backward()
    opt_discriminator.step()
    
    opt_generator.zero_grad()
    generator_loss = model.generator_loss(X_batch)
    generator_loss.backward()
    opt_generator.step()
    
    return discriminator_loss.cpu().item()+generator_loss.cpu().item()

def train_epoch(train_generator, model, gan_optimizer, loss_function):
    epoch_loss = 0
    epoch_len = 0
    for X_batch, y_batch in train_generator:
        batch_len = len(X_batch)
        batch_loss = train_batch(model, X_batch, y_batch, optimizer, loss_function)
        epoch_loss += batch_loss*batch_len
        epoch_len += batch_len
        
    return epoch_loss / epoch_len

def trainer(count_of_epoch, 
            batch_size, 
            dataset,
            model, 
            loss_function,
            optimizer,
            lr = 0.001):
    
    opt_generator = optimizer(model.generator.parameters(), lr=lr*10, betas=(0.5, 0.999))
    opt_discriminator = optimizer(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    iterations = tqdm(range(count_of_epoch), desc='epoch')
    iterations.set_postfix({'epoch loss': np.nan})
    for it in iterations:        
        batch_generator = tqdm()
        
        epoch_loss = train_epoch(
            train_generator=batch_generator, 
            model=model, 
            loss_function=loss_function, 
            gan_optimizer=(opt_generator, opt_discriminator))
        
        iterations.set_postfix({'epoch loss': epoch_loss})